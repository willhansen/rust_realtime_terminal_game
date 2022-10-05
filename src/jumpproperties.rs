// height is in world squares, and duration is in ticks
// this struct is only aware of one length unit and one time unit.
pub struct JumpProperties {
    pub g: f32,
    pub delta_vy: f32,
    pub height: f32,
    pub duration: f32,
}

impl JumpProperties {
    pub fn from_height_and_duration(height: f32, duration: f32) -> JumpProperties {
        let g = JumpProperties::g_from_jump_height_and_duration(height, duration);
        JumpProperties {
            g,
            delta_vy: JumpProperties::jump_vel_from_height_and_g(height, g),
            height,
            duration,
        }
    }

    pub fn from_delta_v_and_g(delta_vy: f32, g: f32) -> JumpProperties {
        JumpProperties {
            g,
            delta_vy,
            height: JumpProperties::height_from_jump_speed_and_g(delta_vy, g),
            duration: JumpProperties::duration_from_jump_speed_and_g(delta_vy, g),
        }
    }

    pub fn height_from_jump_speed_and_g(jump_vel: f32, grav_accel: f32) -> f32 {
        0.5 * jump_vel * jump_vel / grav_accel.abs()
    }

    pub fn jump_vel_from_height_and_g(jump_height: f32, grav_accel: f32) -> f32 {
        (2.0 * grav_accel.abs() * jump_height).sqrt()
    }

    pub fn time_to_jump_peak(jump_vel: f32, grav_accel: f32) -> f32 {
        jump_vel.abs() / grav_accel.abs()
    }

    pub fn duration_from_jump_speed_and_g(jump_vel: f32, grav_accel: f32) -> f32 {
        JumpProperties::time_to_jump_peak(jump_vel, grav_accel) * 2.0
    }

    pub fn g_from_jump_height_and_duration(jump_height: f32, duration: f32) -> f32 {
        let time_to_peak = duration / 2.0;
        2.0 * jump_height / (time_to_peak * time_to_peak)
    }
}
