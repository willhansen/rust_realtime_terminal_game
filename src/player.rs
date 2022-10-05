use crate::{
    jumpproperties, p, AddButInstantTurnAround, BoostBehavior, KinematicState,
    PlayerBlockCollision, Point, SpeedLineType, WallJumpBehavior,
    DEFAULT_PARTICLE_LIFETIME_IN_TICKS, DEFAULT_PLAYER_ACCELERATION_FROM_AIR_TRACTION,
    DEFAULT_PLAYER_ACCELERATION_FROM_FLOOR_TRACTION, DEFAULT_PLAYER_AIR_FRICTION_DECELERATION,
    DEFAULT_PLAYER_AIR_FRICTION_START_SPEED, DEFAULT_PLAYER_DASH_SPEED,
    DEFAULT_PLAYER_GROUND_FRICTION_DECELERATION, DEFAULT_PLAYER_GROUND_FRICTION_START_SPEED,
    DEFAULT_PLAYER_JUMP_DURATION_IN_SECONDS, DEFAULT_PLAYER_JUMP_HEIGHT_IN_GRID_COORDINATES,
    DEFAULT_PLAYER_MAX_RUN_SPEED, DEFAULT_PLAYER_MIDAIR_MAX_MOVE_SPEED,
    ENABLE_JUMP_COMPRESSION_BONUS, VERTICAL_GRID_STRETCH_FACTOR,
};
use jumpproperties::JumpProperties;
use std::collections::VecDeque;

pub struct Player {
    pub alive: bool,
    pub pos: Point<f32>,
    pub recent_kinematic_states: VecDeque<KinematicState>,
    pub max_run_speed: f32,
    pub ground_friction_start_speed: f32,
    pub air_friction_start_speed: f32,
    pub max_midair_move_speed: f32,
    pub speed_of_blue: f32,
    pub vel: Point<f32>,
    pub accel: Point<f32>,
    pub desired_direction: Point<i32>,
    pub jump_properties: JumpProperties,
    pub acceleration_from_floor_traction: f32,
    pub acceleration_from_air_traction: f32,
    pub deceleration_from_air_friction: f32,
    pub deceleration_from_ground_friction: f32,
    pub dash_vel: f32,
    pub boost_behavior: BoostBehavior,
    pub time_of_last_boost: Option<f32>,
    pub last_collision: Option<PlayerBlockCollision>,
    pub moved_normal_to_collision_since_collision: bool,
    pub speed_line_lifetime_in_ticks: f32,
    pub speed_line_behavior: SpeedLineType,
    pub enable_jump_compression_bonus: bool,
    pub wall_jump_behavior: WallJumpBehavior,
}

impl Player {
    pub fn new() -> Player {
        let mut new_player = Player {
            alive: false,
            pos: p(0.0, 0.0),
            recent_kinematic_states: VecDeque::<KinematicState>::new(),
            max_run_speed: DEFAULT_PLAYER_MAX_RUN_SPEED,
            ground_friction_start_speed: DEFAULT_PLAYER_GROUND_FRICTION_START_SPEED,
            air_friction_start_speed: DEFAULT_PLAYER_AIR_FRICTION_START_SPEED,
            max_midair_move_speed: DEFAULT_PLAYER_MIDAIR_MAX_MOVE_SPEED,
            speed_of_blue: DEFAULT_PLAYER_DASH_SPEED,
            vel: Point::<f32>::new(0.0, 0.0),
            accel: Point::<f32>::new(0.0, 0.0),
            desired_direction: p(0, 0),
            jump_properties: JumpProperties::from_height_and_duration(
                DEFAULT_PLAYER_JUMP_HEIGHT_IN_GRID_COORDINATES * VERTICAL_GRID_STRETCH_FACTOR,
                crate::seconds_to_ticks(DEFAULT_PLAYER_JUMP_DURATION_IN_SECONDS),
            ),
            acceleration_from_floor_traction: DEFAULT_PLAYER_ACCELERATION_FROM_FLOOR_TRACTION,
            acceleration_from_air_traction: DEFAULT_PLAYER_ACCELERATION_FROM_AIR_TRACTION,
            deceleration_from_air_friction: DEFAULT_PLAYER_AIR_FRICTION_DECELERATION,
            deceleration_from_ground_friction: DEFAULT_PLAYER_GROUND_FRICTION_DECELERATION,
            dash_vel: DEFAULT_PLAYER_DASH_SPEED,
            boost_behavior: AddButInstantTurnAround,
            time_of_last_boost: None,
            last_collision: None,
            moved_normal_to_collision_since_collision: false,
            speed_line_lifetime_in_ticks: DEFAULT_PARTICLE_LIFETIME_IN_TICKS,
            speed_line_behavior: SpeedLineType::PerpendicularLines,
            enable_jump_compression_bonus: ENABLE_JUMP_COMPRESSION_BONUS,
            wall_jump_behavior: WallJumpBehavior::SwitchDirection,
        };
        new_player
    }

    pub fn kinematic_state(&self) -> KinematicState {
        KinematicState {
            pos: self.pos,
            vel: self.vel,
            accel: self.accel,
        }
    }
}
