#ifndef TRAJECTORY_PLANNING_PROCESSOR_HH_
#define TRAJECTORY_PLANNING_PROCESSOR_HH_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>

inline constexpr std::size_t OBSERVATION_DIM = 186;
inline constexpr std::size_t ACTION_DIM = 6;

/** @class TrajectoryPlanningProcessor
 * @brief Processes actions, observations, gate transitions, and rewards for a
 *        quadrotor trajectory-planning environment.
 *
 * State layout assumed by this class:
 *   [position(3), velocity(3), quaternion(4), angular_velocity(3)]
 *
 * Each stored state-history entry excludes position and therefore has 10 values:
 *   [velocity(3), quaternion(4), angular_velocity(3)]
 *
 * Observation layout:
 *   state history                                      10 * last_state_history_size
 *   relative center of next gate                       3
 *   normal of next gate                                3
 *   velocity toward next gate                          1
 *   normalized distance to next gate                   1
 *   relative corners of next gate                      12
 *   next-next center relative to next gate              3
 *   normal of next-next gate                           3
 *   action history                                     act_dim * action_history_size
 *
 * Gate waypoint formats accepted by this class:
 *   [x, y, z, yaw]                         legacy horizontal heading
 *   [x, y, z, yaw, pitch]                  full 3D heading angles
 *   [x, y, z, nx, ny, nz]                  full 3D heading vector
 *   [x, y, z, nx, ny, nz, ux, uy, uz]      heading plus preferred gate-up
 *
 * For six-value waypoints, the gate-up vector is propagated from the previous
 * gate using projection (a discrete parallel-transport frame). This avoids the
 * singularity of yaw near a vertical heading. Use the nine-value form when the
 * roll of the rectangular aperture must be specified exactly.
 *
 * Gate-radius behavior is explicit:
 *   - FullRectangularOpening automatically enforces radius >= half diagonal.
 *   - CentralCircularCorridor retains the requested smaller center radius.
 *
 * The constructor verifies that these features total OBSERVATION_DIM exactly.
 */
class TrajectoryPlanningProcessor {
 public:
    using State = Eigen::Matrix<float, OBSERVATION_DIM, 1>;
    using Statef = Eigen::Matrix<float, 13, 1>;
    using Action = Eigen::Matrix<float, ACTION_DIM, 1>;

    /** @brief Selects how the circular center-radius constraint is applied.
     *
     * FullRectangularOpening:
     *   The effective crossing radius is automatically clamped to at least
     *       0.5 * hypot(gate_width, gate_height)
     *   so every point inside the rectangular aperture can be accepted.
     *
     * CentralCircularCorridor:
     *   The caller-provided radius is used unchanged. Points must be inside
     *   both the rectangular aperture and this centered circular corridor.
     */
    enum class GateAcceptanceMode {
        FullRectangularOpening,
        CentralCircularCorridor
    };

    TrajectoryPlanningProcessor(
        int action_history_size,
        int last_state_history_size,
        bool xyz,
        bool vxyz,
        const std::vector<std::vector<float>>& gate_centers,
        float gate_width,
        float gate_height,
        float gate_crossing_radius,
        const Eigen::VectorXf& xyz_scaling,
        const Eigen::VectorXf& vxyz_scaling,
        GateAcceptanceMode gate_acceptance_mode =
            GateAcceptanceMode::FullRectangularOpening)
        : action_history_size_(action_history_size),
          last_state_history_size_(last_state_history_size),
          gate_width_(gate_width),
          gate_height_(gate_height),
          gate_crossing_radius_(gate_crossing_radius),
          gate_acceptance_mode_(gate_acceptance_mode),
          xyz_scaling_(xyz_scaling),
          vxyz_scaling_(vxyz_scaling),
          use_xyz_(xyz),
          use_vxyz_(vxyz) {
        ValidateConfiguration(gate_centers);
        minimum_full_opening_radius_ =
            ComputeMinimumFullOpeningRadius(gate_width_, gate_height_);
        if (gate_acceptance_mode_ ==
            GateAcceptanceMode::FullRectangularOpening) {
            gate_crossing_radius_ = std::max(
                gate_crossing_radius_, minimum_full_opening_radius_);
        }

        if (use_xyz_ && use_vxyz_) {
            act_dim_ = 6;
        } else {
            act_dim_ = 3;
        }

        ValidateScalingVectors();

        net_scaling_.resize(act_dim_);
        if (use_xyz_ && use_vxyz_) {
            net_scaling_.head<3>() = xyz_scaling_;
            net_scaling_.tail<3>() = vxyz_scaling_;
        } else if (use_xyz_) {
            net_scaling_ = xyz_scaling_;
        } else {
            net_scaling_ = vxyz_scaling_;
        }

        ValidateObservationDimension();

        num_gates_ = static_cast<int>(gate_centers.size());
        ExtractGateCornersFromWaypoints(gate_centers);
        ComputeAverageGateDistance();

        action_history_.assign(
            static_cast<std::size_t>(action_history_size_ * act_dim_), 0.0f);
        last_state_history_.assign(
            static_cast<std::size_t>(last_state_history_size_ * kStateHistorySize),
            0.0f);
        previous_action_ = Eigen::VectorXf::Zero(act_dim_);
        reward_gate_index_ = 0;
    }

    /** @brief Builds arbitrary-orientation gate frames and corners.
     *
     * Supported waypoint formats:
     *   4: [x, y, z, yaw]
     *   5: [x, y, z, yaw, pitch]
     *   6: [x, y, z, nx, ny, nz]
     *   9: [x, y, z, nx, ny, nz, ux, uy, uz]
     *
     * The 6-value representation is recommended for generated 3D curves.
     * The 9-value representation additionally fixes rotation about the normal.
     */
    void ExtractGateCornersFromWaypoints(
        const std::vector<std::vector<float>>& waypoints) {
        gate_corners_.resize(num_gates_, 12);
        gate_centers_.resize(num_gates_, 3);
        gate_normals_.resize(num_gates_, 3);
        gate_laterals_.resize(num_gates_, 3);
        gate_ups_.resize(num_gates_, 3);
        gate_corners_.setZero();
        gate_centers_.setZero();
        gate_normals_.setZero();
        gate_laterals_.setZero();
        gate_ups_.setZero();

        for (int i = 0; i < num_gates_; ++i) {
            const auto& waypoint = waypoints[static_cast<std::size_t>(i)];
            const std::size_t waypoint_size = waypoint.size();
            if (waypoint_size != 4U && waypoint_size != 5U &&
                waypoint_size != 6U && waypoint_size != 9U) {
                std::ostringstream oss;
                oss << "Waypoint " << i << " has " << waypoint_size
                    << " values; expected [x,y,z,yaw], [x,y,z,yaw,pitch], "
                       "[x,y,z,nx,ny,nz], or "
                       "[x,y,z,nx,ny,nz,ux,uy,uz].";
                throw std::invalid_argument(oss.str());
            }

            for (float value : waypoint) {
                if (!std::isfinite(value)) {
                    std::ostringstream oss;
                    oss << "Waypoint " << i << " contains a non-finite value.";
                    throw std::invalid_argument(oss.str());
                }
            }

            const Eigen::Vector3f center(
                waypoint[0], waypoint[1], waypoint[2]);
            gate_centers_.row(i) = center.transpose();

            Eigen::Vector3f normal = Eigen::Vector3f::Zero();
            Eigen::Vector3f preferred_up = Eigen::Vector3f::Zero();
            bool has_explicit_up = false;

            if (waypoint_size == 4U || waypoint_size == 5U) {
                const float yaw = waypoint[3];
                const float pitch = waypoint_size == 5U ? waypoint[4] : 0.0f;
                const float cos_pitch = std::cos(pitch);
                normal = Eigen::Vector3f(
                    cos_pitch * std::cos(yaw),
                    cos_pitch * std::sin(yaw),
                    std::sin(pitch));
            } else {
                normal = Eigen::Vector3f(
                    waypoint[3], waypoint[4], waypoint[5]);
                if (waypoint_size == 9U) {
                    preferred_up = Eigen::Vector3f(
                        waypoint[6], waypoint[7], waypoint[8]);
                    has_explicit_up = true;
                }
            }

            const float normal_norm = normal.norm();
            if (!std::isfinite(normal_norm) || normal_norm <= kFrameEpsilon) {
                std::ostringstream oss;
                oss << "Waypoint " << i
                    << " has a zero or invalid 3D heading vector.";
                throw std::invalid_argument(oss.str());
            }
            normal = normal / normal_norm;

            Eigen::Vector3f up_candidate = Eigen::Vector3f::Zero();
            if (has_explicit_up) {
                up_candidate = ProjectOntoPlane(preferred_up, normal);
                if (up_candidate.norm() <= kFrameEpsilon) {
                    std::ostringstream oss;
                    oss << "Waypoint " << i
                        << " has an up vector parallel to its heading.";
                    throw std::invalid_argument(oss.str());
                }
            } else if (i > 0) {
                const Eigen::Vector3f previous_up =
                    gate_ups_.row(i - 1).transpose();
                up_candidate = ProjectOntoPlane(previous_up, normal);

                // A 90-degree heading change can collapse that projection.
                // Try the previous lateral before falling back to a world axis.
                if (up_candidate.norm() <= kFrameEpsilon) {
                    const Eigen::Vector3f previous_lateral =
                        gate_laterals_.row(i - 1).transpose();
                    up_candidate = ProjectOntoPlane(previous_lateral, normal);
                }
            }

            if (up_candidate.norm() <= kFrameEpsilon) {
                const Eigen::Vector3f reference = LeastAlignedWorldAxis(normal);
                up_candidate = ProjectOntoPlane(reference, normal);
            }

            float up_norm = up_candidate.norm();
            if (!std::isfinite(up_norm) || up_norm <= kFrameEpsilon) {
                std::ostringstream oss;
                oss << "Could not construct a stable gate frame for waypoint "
                    << i << ".";
                throw std::invalid_argument(oss.str());
            }
            Eigen::Vector3f up = up_candidate / up_norm;

            // For inferred frames, preserve sign continuity between gates.
            if (!has_explicit_up && i > 0) {
                const Eigen::Vector3f previous_up =
                    gate_ups_.row(i - 1).transpose();
                if (up.dot(previous_up) < 0.0f) {
                    up = -up;
                }
            }

            Eigen::Vector3f lateral = Cross(up, normal);
            const float lateral_norm = lateral.norm();
            if (!std::isfinite(lateral_norm) ||
                lateral_norm <= kFrameEpsilon) {
                std::ostringstream oss;
                oss << "Could not construct a gate lateral axis for waypoint "
                    << i << ".";
                throw std::invalid_argument(oss.str());
            }
            lateral = lateral / lateral_norm;

            // Recompute up to remove accumulated projection error and guarantee
            // an orthonormal, right-handed [normal, lateral, up] frame.
            up = Cross(normal, lateral);
            up_norm = up.norm();
            if (up_norm <= kFrameEpsilon) {
                throw std::logic_error("Gate-frame orthogonalization failed.");
            }
            up = up / up_norm;

            gate_normals_.row(i) = normal.transpose();
            gate_laterals_.row(i) = lateral.transpose();
            gate_ups_.row(i) = up.transpose();

            const Eigen::Vector3f half_lateral =
                lateral * (0.5f * gate_width_);
            const Eigen::Vector3f half_up = up * (0.5f * gate_height_);

            const Eigen::Vector3f corners[4] = {
                center - half_lateral + half_up,
                center + half_lateral + half_up,
                center + half_lateral - half_up,
                center - half_lateral - half_up};

            for (int corner = 0; corner < 4; ++corner) {
                for (int axis = 0; axis < 3; ++axis) {
                    gate_corners_(i, 3 * corner + axis) = corners[corner](axis);
                }
            }
        }
    }

    /** @brief Computes mean spacing for observation normalization.
     *
     * The final-to-first segment is included because the waypoint generators
     * produce closed circle/Lissajous paths.
     */
    void ComputeAverageGateDistance() {
        if (num_gates_ < 2) {
            avg_gate_distance_ = 1.0f;
            return;
        }

        float total_distance = 0.0f;
        for (int i = 0; i < num_gates_; ++i) {
            const int next_i = (i + 1) % num_gates_;
            const Eigen::Vector3f current =
                gate_centers_.row(i).transpose();
            const Eigen::Vector3f next =
                gate_centers_.row(next_i).transpose();
            total_distance += (next - current).norm();
        }

        avg_gate_distance_ = total_distance / static_cast<float>(num_gates_);
        if (!std::isfinite(avg_gate_distance_) || avg_gate_distance_ < kEpsilon) {
            avg_gate_distance_ = 1.0f;
        }
    }

    /** @brief Updates the active gate using forward segment-plane crossing.
     *
     * A gate is crossed only if the segment from previous_pos to current_pos:
     *   1. crosses the gate plane from the negative to the positive side;
     *   2. intersects the rectangular gate aperture; and
     *   3. intersects within the effective gate_crossing_radius.
     *
     * In FullRectangularOpening mode, the effective radius is at least the
     * rectangular half-diagonal. In CentralCircularCorridor mode, it is the
     * caller-provided radius and can intentionally clip the rectangle corners.
     */
    void UpdateGateIndex(
        const Eigen::Vector3f& previous_pos,
        const Eigen::Vector3f& current_pos) {
        // Make repeated processing of the same transition idempotent.
        if (transition_valid_ &&
            (previous_pos - transition_previous_pos_).norm() <=
                kTransitionMatchTolerance &&
            (current_pos - transition_current_pos_).norm() <=
                kTransitionMatchTolerance) {
            return;
        }

        reward_gate_index_ = gate_index_;
        gates_crossed_this_step_ = 0;
        transition_previous_pos_ = previous_pos;
        transition_current_pos_ = current_pos;
        transition_valid_ = true;

        gates_crossed_this_step_ = CountSequentialGateCrossings(
            previous_pos, current_pos, gate_index_);

        if (gates_crossed_this_step_ > 0) {
            gates_crossed_ += gates_crossed_this_step_;
            gate_index_ =
                (gate_index_ + gates_crossed_this_step_) % num_gates_;

            const Eigen::Vector3f new_gate_center =
                gate_centers_.row(gate_index_).transpose();
            initial_dist_to_gate_ = (current_pos - new_gate_center).norm();
            dist_to_current_gate_ = initial_dist_to_gate_;
        } else {
            const Eigen::Vector3f gate_center =
                gate_centers_.row(gate_index_).transpose();
            dist_to_current_gate_ = (current_pos - gate_center).norm();
        }
    }

    /** @brief Returns the number of gates. */
    int GateSize() const noexcept { return num_gates_; }

    /** @brief Returns the effective radius used by the crossing test. */
    float GateCrossingRadius() const noexcept {
        return gate_crossing_radius_;
    }

    /** @brief Returns the half-diagonal needed to admit the full rectangle. */
    float MinimumFullOpeningRadius() const noexcept {
        return minimum_full_opening_radius_;
    }

    /** @brief Returns the configured aperture-acceptance mode. */
    GateAcceptanceMode AcceptanceMode() const noexcept {
        return gate_acceptance_mode_;
    }

    /** @brief Returns the number of gates crossed since Reset(). */
    int GetGatesCrossed() const noexcept { return gates_crossed_; }

    /** @brief Returns whether the last processed transition crossed a gate. */
    bool GateCrossedThisStep() const noexcept {
        return gates_crossed_this_step_ > 0;
    }

    /** @brief Returns the number of sequential gates crossed last transition. */
    int GatesCrossedThisStep() const noexcept {
        return gates_crossed_this_step_;
    }

    /** @brief Resets processor state. */
    void Reset(int gate_index) {
        if (gate_index < 0 || gate_index >= num_gates_) {
            std::ostringstream oss;
            oss << "gate_index=" << gate_index << " is outside [0, "
                << (num_gates_ - 1) << "].";
            throw std::out_of_range(oss.str());
        }

        gate_index_ = gate_index;
        reward_gate_index_ = gate_index;
        gates_crossed_ = 0;
        gates_crossed_this_step_ = 0;
        transition_valid_ = false;
        current_pos_.setZero();
        last_velocity_.setZero();

        action_history_.assign(
            static_cast<std::size_t>(action_history_size_ * act_dim_), 0.0f);
        last_state_history_.assign(
            static_cast<std::size_t>(last_state_history_size_ * kStateHistorySize),
            0.0f);

        initial_dist_to_gate_ = avg_gate_distance_;
        dist_to_current_gate_ = avg_gate_distance_;
        previous_action_smoothness_ = 0.0f;
        previous_action_ = Eigen::VectorXf::Zero(act_dim_);
    }

    /** @brief Adds a processed action to the action history. */
    void PushAction(const Eigen::VectorXf& action) {
        if (action.size() != act_dim_) {
            std::ostringstream oss;
            oss << "Action has dimension " << action.size()
                << "; expected " << act_dim_ << ".";
            throw std::invalid_argument(oss.str());
        }

        previous_action_smoothness_ = (action - previous_action_).norm();
        previous_action_ = action;

        for (int i = 0; i < action.size(); ++i) {
            action_history_.push_back(action(i));
            if (static_cast<int>(action_history_.size()) >
                action_history_size_ * act_dim_) {
                action_history_.pop_front();
            }
        }
    }

    /** @brief Stores [velocity, quaternion, angular velocity] from a state. */
    void PushLastState(const Statef& state) {
        for (int i = 3; i < 13; ++i) {
            last_state_history_.push_back(state(i));
            if (static_cast<int>(last_state_history_.size()) >
                last_state_history_size_ * kStateHistorySize) {
                last_state_history_.pop_front();
            }
        }
    }

    /** @brief Converts the current transition into the fixed-size observation. */
    State& ProcessObservation(
        const Statef& current_state,
        const Statef& current_state_dot,
        const Statef& previous_state,
        const Statef& previous_state_dot,
        State& processed_obs) {
        // These are retained in the API for compatibility, but are not part of
        // the current observation definition.
        (void)current_state_dot;
        (void)previous_state_dot;

        processed_obs.setZero();
        int idx = 0;

        auto append = [&](float value) {
            if (idx >= static_cast<int>(OBSERVATION_DIM)) {
                throw std::logic_error(
                    "Observation writer exceeded OBSERVATION_DIM.");
            }
            processed_obs(idx++) = SanitizeValue(value);
        };

        PushLastState(current_state);
        current_pos_ = current_state.head<3>();
        const Eigen::Vector3f velocity = current_state.segment<3>(3);
        last_velocity_ = velocity;

        UpdateGateIndex(previous_state.head<3>(), current_pos_);

        for (float value : last_state_history_) {
            append(value);
        }

        const int next_gate_idx = gate_index_;
        const int next_next_gate_idx = (gate_index_ + 1) % num_gates_;

        const Eigen::Vector3f next_gate_center =
            gate_centers_.row(next_gate_idx).transpose();
        const Eigen::Vector3f relative_next_center =
            next_gate_center - current_pos_;
        for (int axis = 0; axis < 3; ++axis) {
            append(relative_next_center(axis));
        }

        const Eigen::Vector3f next_gate_normal =
            gate_normals_.row(next_gate_idx).transpose();
        for (int axis = 0; axis < 3; ++axis) {
            append(next_gate_normal(axis));
        }

        const float distance_to_gate = relative_next_center.norm();
        Eigen::Vector3f direction_to_gate = Eigen::Vector3f::Zero();
        if (distance_to_gate > kEpsilon) {
            direction_to_gate = relative_next_center / distance_to_gate;
        }
        append(velocity.dot(direction_to_gate));
        append(distance_to_gate / std::max(avg_gate_distance_, 0.1f));

        for (int i = 0; i < 12; ++i) {
            append(gate_corners_(next_gate_idx, i) - current_pos_(i % 3));
        }

        const Eigen::Vector3f next_next_center =
            gate_centers_.row(next_next_gate_idx).transpose();
        const Eigen::Vector3f relative_next_next =
            next_next_center - next_gate_center;
        for (int axis = 0; axis < 3; ++axis) {
            append(relative_next_next(axis));
        }

        const Eigen::Vector3f next_next_normal =
            gate_normals_.row(next_next_gate_idx).transpose();
        for (int axis = 0; axis < 3; ++axis) {
            append(next_next_normal(axis));
        }

        for (float action : action_history_) {
            append(action);
        }

        if (idx != static_cast<int>(OBSERVATION_DIM)) {
            std::ostringstream oss;
            oss << "Observation writer produced " << idx
                << " values; expected " << OBSERVATION_DIM << ".";
            throw std::logic_error(oss.str());
        }
        return processed_obs;
    }

    /** @brief Scales a policy action and appends it to action history. */
    void ProcessAction(
        const Action& policy_action,
        Eigen::VectorXf& processed_action) {
        processed_action.resize(act_dim_);
        for (int i = 0; i < act_dim_; ++i) {
            processed_action(i) = policy_action(i) * net_scaling_(i);
        }
        PushAction(processed_action);
    }

    /** @brief Computes the transition reward.
     *
     * The function supports being called either before or after
     * ProcessObservation() for the same transition. When observation processing
     * has already advanced gate_index_, the stored pre-transition gate is used.
     */
    void ComputeReward(
        const Statef& current_state,
        const Statef& current_state_dot,
        const Statef& previous_state,
        const Statef& previous_state_dot,
        const Action& action,
        float& rewards) {
        (void)action;

        const Eigen::Vector3f position = current_state.head<3>();
        const Eigen::Vector3f previous_position = previous_state.head<3>();
        const Eigen::Vector3f velocity = current_state.segment<3>(3);
        const Eigen::Vector3f acceleration =
            current_state_dot.segment<3>(3);
        const Eigen::Vector3f previous_acceleration =
            previous_state_dot.segment<3>(3);

        const bool transition_matches =
            transition_valid_ &&
            (previous_position - transition_previous_pos_).norm() <=
                kTransitionMatchTolerance &&
            (position - transition_current_pos_).norm() <=
                kTransitionMatchTolerance;

        const int target_gate_index =
            transition_matches ? reward_gate_index_ : gate_index_;
        const int crossings = transition_matches
            ? gates_crossed_this_step_
            : CountSequentialGateCrossings(
                  previous_position, position, target_gate_index);

        const Eigen::Vector3f target_center =
            gate_centers_.row(target_gate_index).transpose();
        const Eigen::Vector3f target_normal =
            gate_normals_.row(target_gate_index).transpose();

        const Eigen::Vector3f to_gate = target_center - position;
        const float distance_to_gate = to_gate.norm();
        const float previous_distance_to_gate =
            (target_center - previous_position).norm();
        Eigen::Vector3f direction_to_gate = Eigen::Vector3f::Zero();
        if (distance_to_gate > kEpsilon) {
            direction_to_gate = to_gate / distance_to_gate;
        }

        float reward = 0.0f;

        // Euclidean progress toward the gate center.
        reward +=
            (previous_distance_to_gate - distance_to_gate) * kDistanceReward;

        // Signed velocity toward the gate: moving away is penalized.
        const float speed_toward_gate = velocity.dot(direction_to_gate);
        reward += speed_toward_gate * kVelocityReward;
        if (speed_toward_gate > 0.0f) {
            reward += speed_toward_gate * kPositiveProgressReward;
        }

        // This is a per-step change in acceleration. It is proportional to
        // jerk only when the environment time step is fixed.
        const float acceleration_change =
            (acceleration - previous_acceleration).norm();
        reward -= acceleration_change * kAccelerationChangePenalty;

        const float speed = velocity.norm();
        if (speed < kStagnationSpeed &&
            distance_to_gate > 2.0f * gate_crossing_radius_) {
            reward -= kStagnationPenalty;
        }

        const Eigen::Vector3f angular_velocity = current_state.tail<3>();
        reward -= angular_velocity.squaredNorm() * kAngularRatePenalty;

        // Near a gate, reward only forward alignment. Reverse traversal is not
        // considered successful.
        if (distance_to_gate < gate_crossing_radius_ && speed > kEpsilon) {
            const float forward_alignment =
                std::max(0.0f, velocity.dot(target_normal) / speed);
            reward += forward_alignment * kNearGateAlignmentReward;
        }

        if (crossings > 0) {
            reward += kGateCrossingBonus * static_cast<float>(crossings);
        }

        reward -= previous_action_smoothness_ * kActionSmoothnessPenalty;

        rewards = std::clamp(
            SanitizeValue(reward), -kRewardClamp, kRewardClamp);
    }

    /** @brief Converts NaN/Inf to finite values. */
    static float SanitizeValue(float value) noexcept {
        if (std::isnan(value)) {
            return 0.0f;
        }
        if (std::isinf(value)) {
            return value > 0.0f ? 1.0e5f : -1.0e5f;
        }
        return value;
    }

 private:
    static constexpr int kStateHistorySize = 10;
    static constexpr int kFixedObservationFeatures = 26;
    static constexpr float kEpsilon = 1.0e-6f;
    static constexpr float kFrameEpsilon = 1.0e-5f;
    static constexpr float kTransitionMatchTolerance = 1.0e-4f;

    static float ComputeMinimumFullOpeningRadius(
        float gate_width,
        float gate_height) noexcept {
        return 0.5f * std::hypot(gate_width, gate_height);
    }

    static Eigen::Vector3f Cross(
        const Eigen::Vector3f& a,
        const Eigen::Vector3f& b) noexcept {
        return Eigen::Vector3f(
            a.y() * b.z() - a.z() * b.y(),
            a.z() * b.x() - a.x() * b.z(),
            a.x() * b.y() - a.y() * b.x());
    }

    static Eigen::Vector3f ProjectOntoPlane(
        const Eigen::Vector3f& vector,
        const Eigen::Vector3f& unit_normal) noexcept {
        return vector - unit_normal * vector.dot(unit_normal);
    }

    static Eigen::Vector3f LeastAlignedWorldAxis(
        const Eigen::Vector3f& unit_normal) noexcept {
        const float ax = std::abs(unit_normal.x());
        const float ay = std::abs(unit_normal.y());
        const float az = std::abs(unit_normal.z());
        if (ax <= ay && ax <= az) {
            return Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        }
        if (ay <= az) {
            return Eigen::Vector3f(0.0f, 1.0f, 0.0f);
        }
        return Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    }

    // Reward coefficients. Keep these centralized so they can be tuned without
    // changing reward logic.
    static constexpr float kDistanceReward = 2.0f;
    static constexpr float kVelocityReward = 0.3f;
    static constexpr float kPositiveProgressReward = 0.1f;
    static constexpr float kAccelerationChangePenalty = 0.05f;
    static constexpr float kStagnationSpeed = 0.1f;
    static constexpr float kStagnationPenalty = 0.1f;
    static constexpr float kAngularRatePenalty = 0.005f;
    static constexpr float kNearGateAlignmentReward = 0.5f;
    static constexpr float kGateCrossingBonus = 5.0f;
    static constexpr float kActionSmoothnessPenalty = 0.1f;
    static constexpr float kRewardClamp = 500.0f;

    void ValidateConfiguration(
        const std::vector<std::vector<float>>& gate_centers) const {
        if (action_history_size_ < 0 || last_state_history_size_ < 0) {
            throw std::invalid_argument("History sizes must be non-negative.");
        }
        if (!use_xyz_ && !use_vxyz_) {
            throw std::invalid_argument(
                "At least one of xyz or vxyz must be enabled.");
        }
        if (gate_centers.empty()) {
            throw std::invalid_argument("At least one gate is required.");
        }
        if (!std::isfinite(gate_width_) || gate_width_ <= 0.0f ||
            !std::isfinite(gate_height_) || gate_height_ <= 0.0f) {
            throw std::invalid_argument(
                "Gate width and height must be finite and positive.");
        }
        if (!std::isfinite(gate_crossing_radius_) ||
            gate_crossing_radius_ <= 0.0f) {
            throw std::invalid_argument(
                "gate_crossing_radius must be finite and positive.");
        }

        switch (gate_acceptance_mode_) {
            case GateAcceptanceMode::FullRectangularOpening:
            case GateAcceptanceMode::CentralCircularCorridor:
                break;
            default:
                throw std::invalid_argument(
                    "gate_acceptance_mode has an invalid enum value.");
        }
    }

    void ValidateScalingVectors() const {
        if (use_xyz_ && xyz_scaling_.size() != 3) {
            throw std::invalid_argument("xyz_scaling must contain 3 values.");
        }
        if (use_vxyz_ && vxyz_scaling_.size() != 3) {
            throw std::invalid_argument("vxyz_scaling must contain 3 values.");
        }

        if (use_xyz_) {
            for (int i = 0; i < xyz_scaling_.size(); ++i) {
                if (!std::isfinite(xyz_scaling_(i))) {
                    throw std::invalid_argument(
                        "xyz_scaling contains a non-finite value.");
                }
            }
        }
        if (use_vxyz_) {
            for (int i = 0; i < vxyz_scaling_.size(); ++i) {
                if (!std::isfinite(vxyz_scaling_(i))) {
                    throw std::invalid_argument(
                        "vxyz_scaling contains a non-finite value.");
                }
            }
        }
    }

    void ValidateObservationDimension() const {
        const int expected_dimension =
            last_state_history_size_ * kStateHistorySize +
            kFixedObservationFeatures +
            action_history_size_ * act_dim_;

        if (expected_dimension != static_cast<int>(OBSERVATION_DIM)) {
            std::ostringstream oss;
            oss << "Observation configuration produces " << expected_dimension
                << " values, but OBSERVATION_DIM is " << OBSERVATION_DIM
                << ". Formula: 10*last_state_history_size + 26 + "
                << act_dim_ << "*action_history_size.";
            throw std::invalid_argument(oss.str());
        }
    }

    int CountSequentialGateCrossings(
        const Eigen::Vector3f& previous_pos,
        const Eigen::Vector3f& current_pos,
        int first_gate_index) const {
        int count = 0;
        int candidate_gate = first_gate_index;

        // Cap at one full lap so duplicate/coincident gates cannot create an
        // infinite loop.
        while (count < num_gates_ &&
               SegmentCrossesGate(
                   previous_pos, current_pos, candidate_gate)) {
            ++count;
            candidate_gate = (candidate_gate + 1) % num_gates_;
        }
        return count;
    }

    bool SegmentCrossesGate(
        const Eigen::Vector3f& previous_pos,
        const Eigen::Vector3f& current_pos,
        int gate_index) const {
        const Eigen::Vector3f center =
            gate_centers_.row(gate_index).transpose();
        const Eigen::Vector3f normal =
            gate_normals_.row(gate_index).transpose();

        const float previous_plane_distance =
            (previous_pos - center).dot(normal);
        const float current_plane_distance =
            (current_pos - center).dot(normal);

        // Only count forward crossings. A stationary point on the plane or a
        // reverse traversal does not count.
        if (!(previous_plane_distance <= 0.0f &&
              current_plane_distance > 0.0f)) {
            return false;
        }

        const float denominator =
            current_plane_distance - previous_plane_distance;
        if (denominator <= kEpsilon) {
            return false;
        }

        const float interpolation =
            -previous_plane_distance / denominator;
        if (interpolation < 0.0f || interpolation > 1.0f) {
            return false;
        }

        const Eigen::Vector3f intersection =
            previous_pos + interpolation * (current_pos - previous_pos);
        const Eigen::Vector3f relative = intersection - center;

        const Eigen::Vector3f lateral =
            gate_laterals_.row(gate_index).transpose();
        const Eigen::Vector3f up =
            gate_ups_.row(gate_index).transpose();
        const float lateral_offset = std::abs(relative.dot(lateral));
        const float height_offset = std::abs(relative.dot(up));
        const float radial_offset = std::sqrt(
            lateral_offset * lateral_offset +
            height_offset * height_offset);

        const bool inside_aperture =
            lateral_offset <= 0.5f * gate_width_ + kEpsilon &&
            height_offset <= 0.5f * gate_height_ + kEpsilon;
        const bool within_center_radius =
            radial_offset <= gate_crossing_radius_ + kEpsilon;

        return inside_aperture && within_center_radius;
    }

    // Configuration
    int action_history_size_;
    int last_state_history_size_;
    int act_dim_ = 0;
    const float gate_width_;
    const float gate_height_;
    float gate_crossing_radius_;
    float minimum_full_opening_radius_ = 0.0f;
    GateAcceptanceMode gate_acceptance_mode_ =
        GateAcceptanceMode::FullRectangularOpening;

    // Gate data
    int num_gates_ = 0;
    Eigen::MatrixXf gate_corners_;
    Eigen::MatrixXf gate_centers_;
    Eigen::MatrixXf gate_normals_;
    Eigen::MatrixXf gate_laterals_;
    Eigen::MatrixXf gate_ups_;
    float avg_gate_distance_ = 1.0f;

    // State tracking
    int gate_index_ = 0;
    int reward_gate_index_ = 0;
    int gates_crossed_ = 0;
    std::deque<float> action_history_;
    std::deque<float> last_state_history_;
    Eigen::Vector3f current_pos_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f last_velocity_ = Eigen::Vector3f::Zero();
    int gates_crossed_this_step_ = 0;
    bool transition_valid_ = false;
    Eigen::Vector3f transition_previous_pos_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f transition_current_pos_ = Eigen::Vector3f::Zero();

    // Progress tracking
    float initial_dist_to_gate_ = 1.0f;
    float dist_to_current_gate_ = 1.0f;
    float previous_action_smoothness_ = 0.0f;
    Eigen::VectorXf previous_action_;
    Eigen::VectorXf xyz_scaling_;
    Eigen::VectorXf vxyz_scaling_;
    Eigen::VectorXf net_scaling_;
    bool use_xyz_;
    bool use_vxyz_;
};

#endif  // TRAJECTORY_PLANNING_PROCESSOR_HH_
