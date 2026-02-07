#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <Eigen/Dense>

#include "legged_rl_deploy/motion/cnpy.h"

class MotionLoader
{
public:
    MotionLoader(std::string motion_file, float dt = 0.02)
    : dt(dt)
    {
        // Check if file ends with .csv
        if (motion_file.length() >= 4 && motion_file.substr(motion_file.length() - 4) == ".csv") {
            load_data_from_csv(motion_file);
        } else {
            load_data_from_npz(motion_file);
        }
        num_frames = dof_positions.size();
        duration = num_frames * dt;

        update(0.0f);
    }

    void load_data_from_npz(const std::string& motion_file)
    {
        cnpy::npz_t npz_data = cnpy::npz_load(motion_file);

        auto body_pos_w  = npz_data["body_pos_w"];   // [frame, body_id, 3]
        auto body_quat_w = npz_data["body_quat_w"];  // [frame, body_id, 4]
        auto joint_pos   = npz_data["joint_pos"];    // [frame, dof]
        auto joint_vel   = npz_data["joint_vel"];    // [frame, dof]

        root_positions.clear();
        root_quaternions.clear();
        dof_positions.clear();
        dof_velocities.clear();

        const size_t num_frames_npz = body_pos_w.shape[0];

        for (size_t i = 0; i < num_frames_npz; i++)
        {
            const size_t body_stride_pos  = body_pos_w.shape[1] * body_pos_w.shape[2];
            const size_t body_stride_quat = body_quat_w.shape[1] * body_quat_w.shape[2];

            Eigen::Vector3f root_pos = Eigen::Vector3f::Map(body_pos_w.data<float>() + i * body_stride_pos);
            root_positions.push_back(root_pos);

            Eigen::Quaternionf quat(
                body_quat_w.data<float>()[i * body_stride_quat + 0], // w
                body_quat_w.data<float>()[i * body_stride_quat + 1], // x
                body_quat_w.data<float>()[i * body_stride_quat + 2], // y
                body_quat_w.data<float>()[i * body_stride_quat + 3]  // z
            );
            root_quaternions.push_back(quat);

            Eigen::VectorXf joint_position(joint_pos.shape[1]);
            for (int j = 0; j < joint_pos.shape[1]; j++) {
                joint_position[j] = joint_pos.data<float>()[i * joint_pos.shape[1] + j];
            }

            Eigen::VectorXf joint_velocity(joint_vel.shape[1]);
            for (int j = 0; j < joint_vel.shape[1]; j++) {
                joint_velocity[j] = joint_vel.data<float>()[i * joint_vel.shape[1] + j];
            }

            dof_positions.push_back(joint_position);
            dof_velocities.push_back(joint_velocity);
        }
    }

    void load_data_from_csv(const std::string& motion_file)
    {
        auto data = load_csv(motion_file);
        
        root_positions.clear();
        root_quaternions.clear();
        dof_positions.clear();
        dof_velocities.clear();

        const size_t num_frames_csv = data.size();
        
        for(size_t i = 0; i < num_frames_csv; ++i)
        {
            if (data[i].size() < 7) {
                throw std::runtime_error("CSV row " + std::to_string(i) + " has insufficient columns (expected at least 7)");
            }
            
            // CSV format: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, joint_1, joint_2, ...]
            root_positions.push_back(Eigen::Vector3f(data[i][0], data[i][1], data[i][2]));
            root_quaternions.push_back(Eigen::Quaternionf(data[i][6], data[i][3], data[i][4], data[i][5]));
            dof_positions.push_back(Eigen::VectorXf::Map(data[i].data() + 7, data[i].size() - 7));
        }
        
        dof_velocities = compute_raw_derivative(dof_positions);
    }

    void update(float time)
    {
        float phase = std::clamp(time, 0.0f, duration);
        float f = phase / dt;
        frame = static_cast<int>(std::floor(f));
        frame = std::min(frame, num_frames - 1);
    }

    Eigen::VectorXf root_position() {
        return root_positions[frame];
    }
    Eigen::Quaternionf root_quaternion() {
        return root_quaternions[frame];
    }
    Eigen::VectorXf joint_pos() {
        return dof_positions[frame];
    }
    Eigen::VectorXf joint_vel() {
        return dof_velocities[frame];
    }

    float dt = 0.02f;
    int num_frames;
    float duration;

    int frame;
    std::vector<Eigen::VectorXf> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;

private:
    std::vector<std::vector<float>> load_csv(const std::string& filename)
    {
        std::vector<std::vector<float>> data;
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Error opening file: " + filename);
        }

        std::string line;
        while (std::getline(file, line))
        {
            std::vector<float> row;
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ','))
            {
                try
                {
                    row.push_back(std::stof(value));
                }
                catch (const std::invalid_argument& e)
                {
                    throw std::runtime_error("Invalid value in CSV: " + value);
                }
            }
            if (!row.empty()) {
                data.push_back(row);
            }
        }
        file.close();
        return data;
    }

    std::vector<Eigen::VectorXf> compute_raw_derivative(const std::vector<Eigen::VectorXf>& data)
    {
        std::vector<Eigen::VectorXf> derivative;
        for(size_t i = 0; i < data.size() - 1; ++i) {
            derivative.push_back((data[i + 1] - data[i]) / dt);
        }
        if (!data.empty()) {
            derivative.push_back(derivative.empty() ? Eigen::VectorXf::Zero(data[0].size()) : derivative.back());
        }
        return derivative;
    }
};
