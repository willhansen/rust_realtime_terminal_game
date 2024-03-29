//#![allow(non_snake_case)]
#![feature(is_sorted)]
#![allow(warnings)]
mod glyph;
mod jumpproperties;
mod player;
mod utility;

extern crate geo;
extern crate line_drawing;
extern crate num;
extern crate std;
extern crate termion;
#[macro_use]
extern crate approx;

use ntest::timeout;
use player::Player;

// use assert2::{assert, check};
use enum_as_inner::EnumAsInner;
use geo::algorithm::euclidean_distance::EuclideanDistance;
use geo::Point;
use num::traits::FloatConst;
use std::char;
use std::cmp::{max, min};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::io::{stdin, stdout, Write};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;
use termion::event::{Event, Key, MouseButton, MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::{IntoRawMode, RawTerminal};

use crate::jumpproperties::JumpProperties;
use crate::BoostBehavior::AddButInstantTurnAround;
use glyph::*;
use utility::*;

// const player_jump_hang_frames: i32 = 4;
const MAX_FPS: f32 = 60.0; // frames per second
const IDEAL_FRAME_DURATION_MS: u128 = (1000.0 / MAX_FPS) as u128;
const PLAYER_COLOR: ColorName = ColorName::Red;
const PLAYER_HIGH_SPEED_COLOR: ColorName = ColorName::Blue;
const NUM_SAVED_PLAYER_KINEMATIC_STATES: i32 = 10;
const NUM_POSITIONS_TO_CHECK_PER_BLOCK_FOR_COLLISIONS: f32 = 8.0;

const DEFAULT_PLAYER_JUMP_HEIGHT_IN_GRID_COORDINATES: f32 = 2.0;
const DEFAULT_PLAYER_JUMP_DURATION_IN_SECONDS: f32 = 0.3;
const DEFAULT_HANGTIME_GRAVITY_MULTIPLIER: f32 = 0.8;
const DEFAULT_HANGTIME_JUMP_SPEED_FRACTION_THRESHOLD: f32 = 0.8;

const VERTICAL_GRID_STRETCH_FACTOR: f32 = 2.0; // because the grid is not really square

//const DEFAULT_PLAYER_ACCELERATION_FROM_GRAVITY: f32 = 0.1;
const DEFAULT_PLAYER_ACCELERATION_FROM_FLOOR_TRACTION: f32 = 0.2;
const DEFAULT_PLAYER_ACCELERATION_FROM_AIR_TRACTION: f32 =
    DEFAULT_PLAYER_ACCELERATION_FROM_FLOOR_TRACTION;
const DEFAULT_PLAYER_GROUND_FRICTION_DECELERATION: f32 =
    DEFAULT_PLAYER_ACCELERATION_FROM_FLOOR_TRACTION; // / 5.0;
const DEFAULT_PLAYER_MAX_RUN_SPEED: f32 = 0.4;
const DEFAULT_PLAYER_GROUND_FRICTION_START_SPEED: f32 = DEFAULT_PLAYER_MAX_RUN_SPEED; //0.7;
const DEFAULT_PLAYER_DASH_SPEED: f32 = DEFAULT_PLAYER_MAX_RUN_SPEED * 5.0;
//const DEFAULT_PLAYER_AIR_FRICTION_DECELERATION: f32 = DEFAULT_PLAYER_GROUND_FRICTION_DECELERATION;
const DEFAULT_PLAYER_AIR_FRICTION_DECELERATION: f32 = 0.0;
const DEFAULT_PLAYER_MIDAIR_MAX_MOVE_SPEED: f32 = DEFAULT_PLAYER_MAX_RUN_SPEED;
const DEFAULT_PLAYER_AIR_FRICTION_START_SPEED: f32 =
    DEFAULT_PLAYER_GROUND_FRICTION_START_SPEED * 3.0;

const DEFAULT_PARTICLE_LIFETIME_IN_SECONDS: f32 = 15.0;
const DEFAULT_PARTICLE_LIFETIME_IN_TICKS: f32 = DEFAULT_PARTICLE_LIFETIME_IN_SECONDS * MAX_FPS;
const DEFAULT_PARTICLE_SPEED: f32 = 0.3;
const DEFAULT_PARTICLE_TURN_RADIANS_PER_TICK: f32 = 0.0;
const DEFAULT_MAX_COMPRESSION: f32 = 0.2;
const DEFAULT_TICKS_TO_MAX_COMPRESSION: f32 = 5.0;
const DEFAULT_TICKS_TO_END_COMPRESSION: f32 = 10.0;
const DEFAULT_TICKS_FROM_MAX_TO_END_COMPRESSION: f32 =
    DEFAULT_TICKS_TO_END_COMPRESSION - DEFAULT_TICKS_TO_MAX_COMPRESSION;
const ENABLE_JUMP_COMPRESSION_BONUS: bool = true;

//const DEFAULT_PARTICLE_DENSITY_FOR_AMALGAMATION: i32 = 6; // just more than a diagonal line

const DEFAULT_PARTICLE_DENSITY_FOR_AMALGAMATION: i32 = 20;
//const DEFAULT_PARTICLES_IN_AMALGAMATION_FOR_EXPLOSION: i32 = DEFAULT_PARTICLE_DENSITY_FOR_AMALGAMATION * 4; // just small enough so the exploded particles don't recombine in adjacent blocks
const DEFAULT_PARTICLES_IN_AMALGAMATION_FOR_EXPLOSION: i32 =
    (DEFAULT_PARTICLE_DENSITY_FOR_AMALGAMATION + 1) * 4; // just large enough so the exploded particles do recombine in adjacent blocks immediately
const DEFAULT_PARTICLES_TO_AMALGAMATION_CHANGE: i32 =
    (DEFAULT_PARTICLES_IN_AMALGAMATION_FOR_EXPLOSION - DEFAULT_PARTICLE_DENSITY_FOR_AMALGAMATION)
        / 4; // 5 sprites over the block's range of particles contained

const RADIUS_OF_EXACTLY_TOUCHING_ZONE: f32 = 0.000001;

const DEFAULT_MINIMUM_PARTICLE_GENERATION_TIME_AFTER_DASH_IN_SECONDS: f32 = 0.5;
const DEFAULT_MINIMUM_PARTICLE_GENERATION_TIME_AFTER_DASH: f32 =
    DEFAULT_MINIMUM_PARTICLE_GENERATION_TIME_AFTER_DASH_IN_SECONDS * MAX_FPS;

// These have no positional information
#[derive(Copy, Clone, PartialEq, Eq, Debug, EnumAsInner)]
enum Block {
    Air,
    Wall,
    Brick,
    ParticleAmalgam(i32),
    Turret,
    StepFoe,
}

impl Block {
    fn character(&self) -> char {
        self.glyph().character
    }
    fn color(&self) -> ColorName {
        self.glyph().fg_color
    }

    fn glyph(&self) -> Glyph {
        match self {
            Block::Air => Glyph::from_char(' '),
            Block::Brick => Glyph::from_char('▪'),
            Block::Wall => Glyph::from_char('█'),
            Block::Turret => Glyph::from_char('⧊'),   //◘▮⏧⌖
            Block::StepFoe => Glyph::from_char('🨅'), //🩑😠
            Block::ParticleAmalgam(num_particles) => {
                let amalgam_stage = num_particles / DEFAULT_PARTICLES_TO_AMALGAMATION_CHANGE;
                match amalgam_stage {
                    0 => Glyph {
                        character: '░',
                        fg_color: ColorName::Blue,
                        bg_color: ColorName::Black,
                    },
                    1 => Glyph {
                        character: '░',
                        fg_color: ColorName::LightBlue,
                        bg_color: ColorName::Black,
                    },
                    2 => Glyph {
                        character: '▒',
                        fg_color: ColorName::LightBlue,
                        bg_color: ColorName::Black,
                    },
                    3 => Glyph {
                        character: '▓',
                        fg_color: ColorName::LightBlue,
                        bg_color: ColorName::Black,
                    },
                    4 => Glyph {
                        character: '█',
                        fg_color: ColorName::Cyan,
                        bg_color: ColorName::Blue,
                    },
                    5 => Glyph {
                        character: '█',
                        fg_color: ColorName::LightCyan,
                        bg_color: ColorName::LightBlue,
                    },
                    _ => Glyph {
                        character: 'X',
                        fg_color: ColorName::Red,
                        bg_color: ColorName::Green,
                    },
                }
            }
        }
    }

    fn subject_to_block_gravity(&self) -> bool {
        // !matches!(self, Block::Air | Block::Wall | Block::ParticleAmalgam(_))
        false
    }
    fn can_collide_with_player(&self) -> bool {
        !matches!(self, Block::Air)
    }
    fn can_be_hit_by_laser(&self) -> bool {
        !matches!(self, Block::Air | Block::Turret)
    }
}

#[derive(Clone, PartialEq, Debug)]
enum ParticleWallCollisionBehavior {
    PassThrough,
    Vanish,
    Bounce,
}
#[derive(EnumIter, PartialEq, Debug, Clone, Copy)]
pub enum WallJumpBehavior {
    SwitchDirection,
    Stop,
    KeepDirection,
}
#[derive(EnumIter, PartialEq, Debug, Clone, Copy)]
pub enum BoostBehavior {
    Add,
    Set,
    AddButInstantTurnAround,
}
#[derive(EnumIter, PartialEq, Debug, Clone, Copy)]
enum InternalCornerBehavior {
    StopPlayer,
    RedirectPlayer,
}

#[derive(Clone, PartialEq, Debug)]
struct Particle {
    ticks_to_expiry: f32,
    pos: Point<f32>,
    prev_pos: Point<f32>,
    start_pos: Point<f32>,
    vel: Point<f32>,
    random_walk_speed: f32,
    wall_collision_behavior: ParticleWallCollisionBehavior,
}

impl Particle {
    fn glyph(&self) -> Glyph {
        Glyph::world_pos_to_colored_braille_glyph(self.pos, ColorName::Blue)
    }

    fn new_at(pos: Point<f32>) -> Particle {
        Particle {
            ticks_to_expiry: f32::INFINITY,
            pos: pos,
            prev_pos: pos,
            start_pos: pos,
            vel: p(0.0, 0.0),
            random_walk_speed: 0.0,
            wall_collision_behavior: ParticleWallCollisionBehavior::Bounce,
        }
    }

    fn in_floating_square(&self, square_center: FPoint, side_length: f32) -> bool {
        let diff = self.pos - square_center;
        let in_x = diff.x().abs() <= side_length / 2.0;
        let in_y = diff.y().abs() <= side_length / 2.0;
        in_x && in_y
    }
}

#[derive(Debug)]
pub struct PlayerBlockCollision {
    time_in_ticks: f32,
    normal: Point<i32>,
    collider_velocity: Point<f32>,
    collider_pos: Point<f32>,
    collided_block_square: Point<i32>,
}

#[derive(PartialEq, Debug)]
pub enum SpeedLineType {
    StillLine,
    BurstChain,
    BurstOnDash,
    PerpendicularLines,
}

#[derive(PartialEq, Debug, Copy, Clone)]
struct Turret {
    square: IPoint,
    laser_direction: FPoint,
    laser_firing_result: Option<SquarecastResult>,
    laser_can_hit_particles: bool,
}

impl Turret {
    fn new() -> Turret {
        Turret {
            square: zero_i(),
            laser_direction: up_f(),
            laser_firing_result: None,
            laser_can_hit_particles: true,
        }
    }
}

#[derive(PartialEq, Debug, Copy, Clone)]
struct StepFoe {
    square: IPoint,
}

struct Game {
    time_from_start_in_ticks: f32,
    recent_tick_durations_s: VecDeque<f32>,
    grid: Vec<Vec<Block>>, // (x,y), left to right, top to bottom
    particle_location_map: ParticleLocationMap,
    output_buffer: Vec<Vec<Glyph>>, // (x,y), left to right, top to bottom
    output_on_screen: Vec<Vec<Glyph>>, // (x,y), left to right, top to bottom
    particles: Vec<Particle>,
    turrets: Vec<Turret>,
    step_foes: Vec<StepFoe>,
    terminal_size: (u16, u16),  // (width, height)
    prev_mouse_pos: (i32, i32), // where mouse was last frame (if pressed)
    running: bool,              // set false to quit
    selected_block: Block,      // What the mouse places
    num_positions_per_block_to_check_for_collisions: f32,
    is_bullet_time: bool,
    bullet_time_factor: f32,
    player: Player,
    particle_amalgamation_density: i32,
    particle_rotation_speed_towards_player: f32,
    internal_corner_behavior: InternalCornerBehavior,
}

impl Game {
    fn new(width: u16, height: u16) -> Game {
        Game {
            time_from_start_in_ticks: 0.0,
            recent_tick_durations_s: VecDeque::<f32>::new(),
            grid: vec![vec![Block::Air; height as usize]; width as usize],
            particle_location_map: HashMap::<Point<i32>, Vec<usize>>::new(),
            output_buffer: vec![vec![Glyph::from_char(' '); height as usize]; width as usize],
            output_on_screen: vec![vec![Glyph::from_char('x'); height as usize]; width as usize],
            particles: Vec::<Particle>::new(),
            turrets: Vec::<Turret>::new(),
            step_foes: Vec::<StepFoe>::new(),
            terminal_size: (width, height),
            prev_mouse_pos: (1, 1),
            running: true,
            selected_block: Block::Wall,
            num_positions_per_block_to_check_for_collisions:
                NUM_POSITIONS_TO_CHECK_PER_BLOCK_FOR_COLLISIONS,
            is_bullet_time: false,
            bullet_time_factor: 0.1,
            player: Player::new(),
            particle_amalgamation_density: DEFAULT_PARTICLE_DENSITY_FOR_AMALGAMATION,
            particle_rotation_speed_towards_player: DEFAULT_PARTICLE_TURN_RADIANS_PER_TICK,
            internal_corner_behavior: InternalCornerBehavior::RedirectPlayer,
        }
    }

    fn time_in_ticks(&self) -> f32 {
        return self.time_from_start_in_ticks;
    }

    fn time_since(&self, t: f32) -> f32 {
        return self.time_in_ticks() - t;
    }

    fn ticks_since_last_player_collision(&self) -> Option<f32> {
        if self.player.last_collision.is_some() {
            Some(self.time_since(self.player.last_collision.as_ref().unwrap().time_in_ticks))
        } else {
            None
        }
    }

    fn set_player_jump_delta_v(&mut self, delta_v: f32) {
        self.player.jump_properties =
            JumpProperties::from_delta_v_and_g(delta_v, self.player.jump_properties.g);
    }

    fn set_jump_duration_in_seconds_and_height_in_grid_squares(
        &mut self,
        duration_in_seconds: f32,
        height_in_grid_squares: f32,
    ) {
        let duration_in_ticks = seconds_to_ticks(duration_in_seconds);
        let height_in_world_squares = height_in_grid_squares * VERTICAL_GRID_STRETCH_FACTOR;
        self.player.jump_properties =
            JumpProperties::from_height_and_duration(height_in_world_squares, duration_in_ticks);
    }

    fn set_player_max_run_speed(&mut self, speed: f32) {
        self.player.max_run_speed = speed;
    }

    fn screen_to_world(&self, terminal_position: &(u16, u16)) -> (i32, i32) {
        // terminal indexes from 1, and the y axis goes top to bottom
        (
            terminal_position.0 as i32 - 1,
            self.terminal_size.1 as i32 - terminal_position.1 as i32,
        )
    }

    fn world_to_screen(&self, world_position: &(i32, i32)) -> (u16, u16) {
        // terminal indexes from 1, and the y axis goes top to bottom
        // world indexes from 0, origin at bottom left
        (
            world_position.0 as u16 + 1,
            (self.terminal_size.1 as i32 - world_position.1) as u16,
        )
    }
    fn get_block(&self, square: Point<i32>) -> Block {
        if !self.square_is_in_world(square) {
            panic!("square {:?} is not in world", square);
        }
        return self.grid[square.x() as usize][square.y() as usize];
    }
    fn try_get_block(&self, square: Point<i32>) -> Option<Block> {
        return if self.square_is_in_world(square) {
            Some(self.grid[square.x() as usize][square.y() as usize])
        } else {
            None
        };
    }
    fn set_block(&mut self, pos: Point<i32>, block: Block) {
        self.grid[pos.x() as usize][pos.y() as usize] = block;
    }

    fn place_line_of_blocks(&mut self, pos0: (i32, i32), pos1: (i32, i32), block: Block) {
        for (x1, y1) in line_drawing::Bresenham::new(pos0, pos1) {
            self.grid[x1 as usize][y1 as usize] = block;
        }
    }

    fn place_boundary_wall(&mut self) {
        let xmax = self.grid.len() as i32 - 1;
        let ymax = self.grid[0].len() as i32 - 1;
        let xmin = 0;
        let ymin = 0;
        self.place_line_of_blocks((xmin, ymin), (xmax, ymin), Block::Wall);
        self.place_line_of_blocks((xmax, ymin), (xmax, ymax), Block::Wall);
        self.place_line_of_blocks((xmax, ymax), (xmin, ymax), Block::Wall);
        self.place_line_of_blocks((xmin, ymax), (xmin, ymin), Block::Wall);
    }

    fn place_wall_rect(&mut self, corner1: IPoint, corner2: IPoint) {
        let xmin = min(corner1.x(), corner2.x());
        let xmax = max(corner1.x(), corner2.x());
        let ymin = min(corner1.y(), corner2.y());
        let ymax = max(corner1.y(), corner2.y());

        for y in ymin..=ymax {
            self.place_line_of_blocks((xmin, y), (xmax, y), Block::Wall);
        }
    }

    fn place_block(&mut self, pos: Point<i32>, block: Block) {
        if !self.square_is_in_world(pos) {
            println!("tried placing block out of world: {:?}", pos);
            return;
        }
        self.grid[pos.x() as usize][pos.y() as usize] = block;
    }

    fn place_wall_block(&mut self, pos: Point<i32>) {
        self.place_block(pos, Block::Wall);
    }

    #[allow(dead_code)]
    fn place_line_of_particles_with_velocity(
        &mut self,
        start: Point<f32>,
        end: Point<f32>,
        vel: Point<f32>,
    ) {
        let mut line = self.make_line_of_particles(start, end);
        for mut p in &mut line {
            p.vel = vel;
        }
        self.add_particles(line);
    }

    fn place_line_of_particles(&mut self, start: Point<f32>, end: Point<f32>) {
        let line = self.make_line_of_particles(start, end);
        self.add_particles(line);
    }

    fn braille_bresenham_line_points(
        start_pos: Point<f32>,
        end_pos: Point<f32>,
    ) -> Vec<Point<f32>> {
        let braille_pos0 = Glyph::world_pos_to_braille_pos(start_pos);
        let braille_pos1 = Glyph::world_pos_to_braille_pos(end_pos);

        line_drawing::Bresenham::new(
            snap_to_grid(braille_pos0).x_y(),
            snap_to_grid(braille_pos1).x_y(),
        )
        .map(|(x, y)| Glyph::braille_pos_to_world_pos(p(x as f32, y as f32)))
        .collect()
    }

    fn make_line_of_particles(&mut self, pos0: Point<f32>, pos1: Point<f32>) -> Vec<Particle> {
        let mut new_particles = Vec::<Particle>::new();
        for particle_pos in Game::braille_bresenham_line_points(pos0, pos1) {
            new_particles.push(Particle::new_at(particle_pos));
        }
        new_particles
    }

    fn place_particle(&mut self, pos: Point<f32>) {
        self.place_particle_with_lifespan(pos, DEFAULT_PARTICLE_LIFETIME_IN_TICKS)
    }
    fn place_n_particles(&mut self, num_particles: i32, pos: Point<f32>) {
        for _ in 0..num_particles {
            self.place_particle_with_lifespan(pos, DEFAULT_PARTICLE_LIFETIME_IN_TICKS)
        }
    }

    fn add_particle(&mut self, p: Particle) {
        self.particles.push(p);
    }

    fn add_particles(&mut self, mut ps: Vec<Particle>) {
        self.particles.append(&mut ps);
    }

    fn place_particle_with_velocity_and_lifetime(
        &mut self,
        pos: Point<f32>,
        vel: Point<f32>,
        life_duration: f32,
    ) {
        let mut particle = self.make_particle_with_velocity(pos, vel);
        particle.ticks_to_expiry = life_duration;
        self.particles.push(particle);
    }

    fn make_particle_with_velocity(&mut self, pos: Point<f32>, vel: Point<f32>) -> Particle {
        let mut particle = Particle::new_at(pos);
        particle.vel = vel;
        particle
    }

    fn place_particle_with_velocity(&mut self, pos: Point<f32>, vel: Point<f32>) {
        let p = self.make_particle_with_velocity(pos, vel);
        self.add_particle(p);
    }

    fn place_particle_with_lifespan(&mut self, pos: Point<f32>, lifespan_in_ticks: f32) {
        let mut particle = Particle::new_at(pos);
        particle.ticks_to_expiry = lifespan_in_ticks;
        self.particles.push(particle);
    }

    fn get_indexes_of_particles_in_square(&self, square: Point<i32>) -> Vec<usize> {
        self.particles
            .iter()
            .enumerate()
            .filter(|(_, particle)| snap_to_grid(particle.pos) == square)
            .map(|(index, _)| index)
            .collect()
    }

    fn count_braille_dots_in_square(&self, square: Point<i32>) -> i32 {
        return if self.square_is_in_world(square) {
            Glyph::count_braille_dots(self.get_buffered_glyph(square).character)
        } else {
            0
        };
    }

    fn draw_visual_braille_point(&mut self, pos: Point<f32>, color: ColorName) {
        self.draw_visual_braille_line(pos, pos, color);
    }

    fn draw_visual_braille_line(
        &mut self,
        start_pos: Point<f32>,
        end_pos: Point<f32>,
        color: ColorName,
    ) {
        let squares_to_place =
            Glyph::get_glyphs_for_colored_braille_line(start_pos, end_pos, color);

        let start_grid_square = snap_to_grid(start_pos);
        let end_grid_square = snap_to_grid(end_pos);
        let bottom_square_y = min(start_grid_square.y(), end_grid_square.y());
        let left_square_x = min(start_grid_square.x(), end_grid_square.x());

        for i in 0..squares_to_place.len() {
            for j in 0..squares_to_place[0].len() {
                if let Some(new_glyph) = squares_to_place[i][j] {
                    let grid_square = p(left_square_x + i as i32, bottom_square_y + j as i32);
                    if !self.square_is_in_world(grid_square) {
                        continue;
                    }
                    let grid_glyph =
                        &mut self.output_buffer[grid_square.x() as usize][grid_square.y() as usize];
                    if Glyph::is_braille(grid_glyph.character) {
                        let combined_braille =
                            Glyph::add_braille(grid_glyph.character, new_glyph.character);
                        *grid_glyph = new_glyph;
                        grid_glyph.character = combined_braille;
                    } else {
                        *grid_glyph = new_glyph;
                    }
                }
            }
        }
    }

    fn clear(&mut self) {
        let (width, height) = termion::terminal_size().unwrap();
        self.grid = vec![vec![Block::Air; height as usize]; width as usize];
    }

    fn place_player(&mut self, x: f32, y: f32) {
        if self.player.alive {
            self.kill_player();
        }
        self.player.vel = p(0.0, 0.0);
        self.player.desired_direction = p(0, 0);
        self.player.pos = p(x, y);
        self.player.alive = true;
    }

    fn place_turret(&mut self, square: IPoint) {
        //if !self.square_is_empty(square) { panic!("Tried to place turret in occupied square: {:?}", square); }
        let mut turret = Turret::new();
        turret.square = square;
        self.turrets.push(turret);
        self.set_block(square, Block::Turret);
    }
    fn place_step_foe(&mut self, square: IPoint) {
        if !self.square_is_empty(square) {
            panic!("Tried to place step foe in occupied square: {:?}", square);
        }
        let mut step_foe = StepFoe { square };
        self.step_foes.push(step_foe);
        self.set_block(square, Block::StepFoe);
    }
    fn player_can_wall_jump(&mut self) -> bool {
        self.player_is_grabbing_wall() || self.player_is_running_up_wall()
    }

    fn player_jump(&mut self) {
        let compression_bonus = self.jump_bonus_vel_from_compression();

        if self.player_can_wall_jump() {
            if let Some(wall_direction) = self.get_lone_touched_wall_direction() {
                self.player
                    .vel
                    .add_assign(floatify(-wall_direction) * self.player.max_run_speed);
                self.player.desired_direction = match self.player.wall_jump_behavior {
                    WallJumpBehavior::SwitchDirection => -wall_direction,
                    WallJumpBehavior::Stop => zero_i(),
                    WallJumpBehavior::KeepDirection => wall_direction,
                };
            }
        }
        self.player
            .vel
            .add_assign(p(0.0, self.player.jump_properties.delta_vy));

        self.player.vel.add_assign(compression_bonus);
    }

    fn player_is_exactly_on_square(&self) -> bool {
        self.pos_is_exactly_on_square(self.player.pos)
    }
    fn pos_is_exactly_on_square(&self, pos: FPoint) -> bool {
        pos == floatify(snap_to_grid(pos))
    }

    fn try_get_player_square_adjacency(&self) -> Option<LocalBlockOccupancy> {
        self.get_square_adjacency(self.player_square())
    }

    fn is_deep_corner(&self, square: IPoint) -> bool {
        if let Some(adjacency) = self.get_square_adjacency(square) {
            is_deep_concave_corner(adjacency)
        } else {
            false
        }
    }
    fn pos_is_in_deep_corner(&self, pos: FPoint) -> bool {
        self.is_deep_corner(snap_to_grid(pos))
    }

    fn player_can_jump(&self) -> bool {
        self.player_is_supported()
            || self.player_is_grabbing_wall()
            || self.player_is_running_up_wall()
    }

    fn player_jump_if_possible(&mut self) {
        if self.player_can_jump() {
            self.player_jump();
        }
    }

    fn player_dash(&mut self) {
        if self.player.desired_direction != p(0, 0) {
            let dash_vel = floatify(self.player.desired_direction) * self.player.dash_vel;
            match self.player.boost_behavior {
                BoostBehavior::Add => self.player.vel.add_assign(dash_vel),
                BoostBehavior::Set => self.player.vel = dash_vel,
                AddButInstantTurnAround => {
                    let want_increase_speed = dash_vel.dot(self.player.vel) > 0.0;
                    if want_increase_speed {
                        self.player.vel.add_assign(dash_vel);
                    } else {
                        self.player.vel = dash_vel;
                    }
                }
            }
            self.player.time_of_last_boost = Some(self.time_in_ticks());
        }
    }
    fn toggle_bullet_time(&mut self) {
        self.is_bullet_time = !self.is_bullet_time;
    }

    fn handle_event(&mut self, evt: Event) {
        match evt {
            Event::Key(ke) => match ke {
                Key::Char('q') => self.running = false,
                Key::Char('1') => self.selected_block = Block::Air,
                Key::Char('2') => self.selected_block = Block::Wall,
                Key::Char('3') => self.selected_block = Block::Brick,
                Key::Char('c') => self.particles.clear(),
                Key::Char('k') => self.kill_player(),
                Key::Char('r') => self.place_player(
                    self.terminal_size.0 as f32 / 2.0,
                    self.terminal_size.1 as f32 / 2.0,
                ),
                Key::Char(' ') => self.player_jump_if_possible(),
                Key::Char('f') => self.player_dash(),
                Key::Char('g') => self.toggle_bullet_time(),
                Key::Char('w') | Key::Up => self.player.desired_direction = up_i(),
                Key::Char('a') | Key::Left => self.player.desired_direction = left_i(),
                Key::Char('s') | Key::Down => self.player.desired_direction = down_i(),
                Key::Char('d') | Key::Right => self.player.desired_direction = right_i(),
                _ => {}
            },
            Event::Mouse(me) => match me {
                MouseEvent::Press(MouseButton::Left, term_x, term_y) => {
                    let (x, y) = self.screen_to_world(&(term_x, term_y));
                    self.place_block(p(x, y), self.selected_block);
                    self.prev_mouse_pos = (x, y);
                }
                MouseEvent::Press(MouseButton::Right, term_x, term_y) => {
                    let (x, y) = self.screen_to_world(&(term_x, term_y));
                    self.place_player(x as f32, y as f32);
                }
                MouseEvent::Hold(term_x, term_y) => {
                    let (x, y) = self.screen_to_world(&(term_x, term_y));
                    self.place_line_of_blocks(self.prev_mouse_pos, (x, y), self.selected_block);
                    self.prev_mouse_pos = (x, y);
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn apply_physics(&mut self, dt_in_ticks: f32) {
        self.apply_gravity_to_blocks();
        if self.player.alive {
            self.update_player_accelerations();
            self.apply_player_kinematics(dt_in_ticks);
            if self.player_is_in_boost() {
                self.generate_speed_particles();
            }
        }
        self.apply_particle_physics(dt_in_ticks);
        self.apply_turret_physics(dt_in_ticks);
    }

    fn apply_physics_in_steps(&mut self, time_span_in_ticks: f32, time_step_in_ticks: f32) {
        // has one smaller step at the end if applicable
        let num_full_steps = (time_span_in_ticks / time_step_in_ticks).floor() as i32;
        for _ in 0..num_full_steps {
            self.apply_physics(time_step_in_ticks);
        }

        let time_covered_by_full_steps = num_full_steps as f32 * time_step_in_ticks;
        let partial_step_length = (time_span_in_ticks - time_covered_by_full_steps).max(0.0);
        if partial_step_length > 0.0 {
            self.apply_physics(partial_step_length);
        }
    }

    fn apply_physics_in_n_steps(&mut self, time_span_in_ticks: f32, n: i32) {
        self.apply_physics_in_steps(time_span_in_ticks, time_span_in_ticks / (n as f32));
    }

    fn get_time_factor(&self) -> f32 {
        return if self.is_bullet_time {
            self.bullet_time_factor
        } else {
            1.0
        };
    }

    fn tick_physics(&mut self) {
        let dt = self.get_time_factor();
        self.time_from_start_in_ticks += dt;
        self.apply_physics(dt);
    }
    fn tick_turrets(&mut self) {
        self.apply_turret_physics(self.get_time_factor());
    }

    fn apply_turret_physics(&mut self, dt_in_ticks: f32) {
        let mut particles_to_destroy = Vec::<usize>::new();
        self.update_particle_location_map();
        for turret_index in 0..self.turrets.len() {
            let mut updated_turret = self.turrets[turret_index].clone();
            let rotation_speed_degrees_per_tick = CLOCKWISE * 2.0;
            let rotation_degrees_this_tick = rotation_speed_degrees_per_tick * dt_in_ticks;
            updated_turret.laser_direction =
                rotated_degrees(updated_turret.laser_direction, rotation_degrees_this_tick);
            let laser_result = self.fire_turret_laser(&updated_turret);
            if let Some(hit_particle_index) = laser_result.collided_particle_index {
                particles_to_destroy.push(hit_particle_index);
            }
            if let Some(hit_square) = laser_result.collided_block_square {
                match self.get_block(hit_square) {
                    Block::ParticleAmalgam(_) => self.decay_amalgam_once(hit_square),
                    _ => {}
                }
            }
            updated_turret.laser_firing_result = Some(laser_result);
            self.turrets[turret_index] = updated_turret;
        }
        self.delete_particles_at_indexes(particles_to_destroy);
    }

    fn fire_turret_laser(&self, turret: &Turret) -> SquarecastResult {
        let laser_start_point = floatify(turret.square) + up_f() * 0.45;
        let turret_range = 20.0;
        let relative_endpoint_in_world_coordinates =
            direction(turret.laser_direction) * turret_range;
        let relative_endpoint_in_grid_coordinates = world_space_to_grid_space(
            relative_endpoint_in_world_coordinates,
            VERTICAL_GRID_STRETCH_FACTOR,
        );
        self.fire_laser(
            laser_start_point,
            laser_start_point + relative_endpoint_in_grid_coordinates,
        )
    }

    fn fire_laser(
        &self,
        start_point_in_grid_coordinates: FPoint,
        end_point_in_grid_coordinates: FPoint,
    ) -> SquarecastResult {
        self.linecast_laser(
            start_point_in_grid_coordinates,
            end_point_in_grid_coordinates,
        )
    }

    fn tick_particles(&mut self) {
        self.apply_particle_physics(self.get_time_factor());
    }

    fn apply_particle_physics(&mut self, dt_in_ticks: f32) {
        self.apply_particle_lifetimes(dt_in_ticks);

        if self.player.alive {
            self.turn_particles_towards_player(dt_in_ticks);
        }

        self.apply_particle_velocities(dt_in_ticks);

        self.delete_out_of_bounds_particles();

        self.combine_dense_particles();
        self.explode_overfull_particle_amalgams();
    }

    fn get_particle_location_map(&self) -> &ParticleLocationMap {
        &self.particle_location_map
    }

    fn update_particle_location_map(&mut self) {
        let mut histogram: ParticleLocationMap = HashMap::<Point<i32>, Vec<usize>>::new();
        for i in 0..self.particles.len() {
            let square = snap_to_grid(self.particles[i].pos);
            match histogram.entry(square) {
                Entry::Vacant(e) => {
                    e.insert(vec![i]);
                }
                Entry::Occupied(mut e) => {
                    e.get_mut().push(i);
                }
            }
        }
        self.particle_location_map = histogram;
    }

    fn combine_dense_particles(&mut self) {
        let mut particle_indexes_to_delete = vec![];
        self.update_particle_location_map();
        let particle_map_copy = self.get_particle_location_map().clone();
        for (square, indexes_of_particles_in_square) in particle_map_copy {
            let indexes_of_particles_that_did_not_start_here: Vec<usize> =
                indexes_of_particles_in_square
                    .iter()
                    .filter(|&&index| snap_to_grid(self.particles[index].start_pos) != square)
                    .cloned()
                    .collect();
            //if !self.square_is_in_world(square) { continue; }
            let block = self.get_block(square);
            let enough_non_native_particles_to_condense =
                indexes_of_particles_that_did_not_start_here.len()
                    >= self.particle_amalgamation_density as usize;
            let block_already_condensed = matches!(block, Block::ParticleAmalgam(_));
            if block_already_condensed {
                let existing_count: i32 = *block.as_particle_amalgam().unwrap();
                self.set_block(
                    square,
                    Block::ParticleAmalgam(
                        indexes_of_particles_that_did_not_start_here.len() as i32 + existing_count,
                    ),
                );
                particle_indexes_to_delete
                    .append(&mut indexes_of_particles_that_did_not_start_here.clone());
            } else if enough_non_native_particles_to_condense {
                self.set_block(
                    square,
                    Block::ParticleAmalgam(indexes_of_particles_in_square.len() as i32),
                );
                particle_indexes_to_delete.append(&mut indexes_of_particles_in_square.clone());
            }
        }
        self.delete_particles_at_indexes(particle_indexes_to_delete);
    }

    fn get_num_particles_in_amalgam(&self, square: IPoint) -> Option<i32> {
        if let Some(block) = self.try_get_block(square) {
            if let Block::ParticleAmalgam(particle_count) = block {
                Some(particle_count)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn remove_particle_from_amalgam(&mut self, square: IPoint) -> bool {
        if let Some(particle_count) = self.get_num_particles_in_amalgam(square) {
            if particle_count > 1 {
                self.set_block(square, Block::ParticleAmalgam(particle_count - 1));
            } else {
                self.set_block(square, Block::Air);
            }
            true
        } else {
            false
        }
    }

    fn decay_amalgam_once(&mut self, square: IPoint) {
        if self.remove_particle_from_amalgam(square) {
            self.place_decay_particle(square);
        }
    }

    fn place_decay_particle(&mut self, square: IPoint) {
        let dir = random_direction();
        let pos = rand_in_square(square);
        let max_speed = 0.5;
        let min_speed = max_speed / 2.0;
        let speed = rand_in_range(min_speed, max_speed);
        self.place_particle_with_velocity_and_lifetime(
            pos,
            dir * speed,
            self.player.speed_line_lifetime_in_ticks,
        );
    }

    fn explode_overfull_particle_amalgams(&mut self) {
        for x in 0..self.width() {
            for y in 0..self.height() {
                let pos = p(x as i32, y as i32);
                if let Some(Block::ParticleAmalgam(num_particles)) = self.try_get_block(pos) {
                    if num_particles >= DEFAULT_PARTICLES_IN_AMALGAMATION_FOR_EXPLOSION {
                        //self.place_grid_space_particle_burst(
                        self.place_particle_blast(
                            floatify(pos),
                            num_particles,
                            DEFAULT_PARTICLE_SPEED,
                        );
                        self.set_block(pos, Block::Air);
                    }
                }
            }
        }
    }

    fn delete_particles_at_indexes(&mut self, mut indexes: Vec<usize>) {
        indexes.sort_unstable();
        indexes.dedup();
        indexes.reverse();

        for i in indexes {
            self.particles.remove(i);
        }
    }

    fn turn_particles_towards_player(&mut self, dt_in_ticks: f32) {
        for i in 0..self.particles.len() {
            let particle = &mut self.particles[i];
            let vect_to_player = self.player.pos - particle.pos;
            if particle.vel != zero_f() && vect_to_player != zero_f() {
                particle.vel = rotate_vector_towards(
                    particle.vel,
                    vect_to_player,
                    self.particle_rotation_speed_towards_player * dt_in_ticks,
                );
            }
        }
    }

    fn apply_particle_velocities(&mut self, dt_in_ticks: f32) {
        let mut particle_indexes_to_delete = vec![];
        for i in 0..self.particles.len() {
            let particle = self.particles[i].clone();

            let mut step = world_space_to_grid_space(
                particle.vel * dt_in_ticks
                    + direction(random_direction())
                        * particle.random_walk_speed
                        * dt_in_ticks.sqrt(),
                VERTICAL_GRID_STRETCH_FACTOR,
            );
            let mut start_pos = particle.pos;
            let mut end_pos = start_pos + step;

            if particle.wall_collision_behavior != ParticleWallCollisionBehavior::PassThrough {
                while magnitude(step) > 0.00001 {
                    let collision = self.linecast_walls_only(start_pos, end_pos);
                    if collision.hit_something() {
                        if particle.wall_collision_behavior == ParticleWallCollisionBehavior::Bounce
                        {
                            let new_start = collision.collider_pos;
                            let step_taken = new_start - start_pos;
                            step.add_assign(-step_taken);
                            let vel = self.particles[i].vel;
                            if collision.collision_normal.unwrap().x() != 0 {
                                step.set_x(-step.x());
                                self.particles[i].vel.set_x(-vel.x());
                            } else {
                                step.set_y(-step.y());
                                self.particles[i].vel.set_y(-vel.y());
                            }
                            start_pos = new_start;
                            end_pos = start_pos + step;
                            //break;
                        } else if particle.wall_collision_behavior
                            == ParticleWallCollisionBehavior::Vanish
                        {
                            particle_indexes_to_delete.push(i);
                            break;
                        }
                    } else {
                        break;
                    }
                }
                let end_square = snap_to_grid(end_pos);
                let particle_ended_inside_wall = self.try_get_block(end_square)
                    == Some(Block::Wall)
                    && point_inside_grid_square(end_pos, end_square);
                if particle_ended_inside_wall
                    && particle.wall_collision_behavior == ParticleWallCollisionBehavior::Bounce
                {
                    dbg!(&self.particles[i], start_pos, end_pos);
                    panic!("particle ended inside wall")
                }
            }

            self.particles[i].prev_pos = self.particles[i].pos;
            self.particles[i].pos = end_pos;
        }
        self.delete_particles_at_indexes(particle_indexes_to_delete);
    }

    fn apply_particle_lifetimes(&mut self, dt_in_ticks: f32) {
        self.particles.iter_mut().for_each(|particle| {
            particle.ticks_to_expiry -= dt_in_ticks;
        });

        self.particles
            .retain(|particle| particle.ticks_to_expiry > 0.0);
    }

    fn delete_out_of_bounds_particles(&mut self) {
        let particles_are_in_world: Vec<bool> = self
            .particles
            .iter()
            .map(|particle| self.square_is_in_world(snap_to_grid(particle.pos)))
            .collect();

        particles_are_in_world
            .into_iter()
            .enumerate()
            .filter(|(_, is_in_world)| !*is_in_world)
            .map(|(i, _)| i)
            .rev()
            .for_each(|i| {
                self.particles.remove(i);
            });
    }

    fn get_player_color(&self) -> ColorName {
        if self.player_is_in_boost() {
            PLAYER_HIGH_SPEED_COLOR
        } else {
            PLAYER_COLOR
        }
    }

    fn jump_bonus_vel_from_compression(&self) -> FPoint {
        if !self.player.enable_jump_compression_bonus || !self.player_is_in_compression() {
            return zero_f();
        }
        if self.player.last_collision.is_none() {
            panic!("Player in compression, but has no last collision");
        }
        let collision = &self.player.last_collision.as_ref().unwrap();
        let g = self.player.jump_properties.g;
        let normal_impact_vel =
            project_a_onto_b(-collision.collider_velocity, floatify(collision.normal));
        let stored_jump_height_in_world_coordinates =
            JumpProperties::height_from_jump_speed_and_g(magnitude(normal_impact_vel), g);
        let speed_for_adding_stored_height = JumpProperties::jump_vel_from_height_and_g(
            stored_jump_height_in_world_coordinates + self.player.jump_properties.height,
            g,
        );
        let speed_to_add = speed_for_adding_stored_height - self.player.jump_properties.delta_vy;
        return direction(normal_impact_vel) * speed_to_add;
    }

    fn get_player_is_coming_out_of_compression(&self) -> bool {
        self.player_is_in_compression()
            && self.ticks_since_last_player_collision().unwrap() > DEFAULT_TICKS_TO_MAX_COMPRESSION
    }

    fn player_is_in_compression(&self) -> bool {
        self.get_player_compression_fraction() < 1.0
    }

    fn get_player_compression_fraction(&self) -> f32 {
        if self.player.moved_normal_to_collision_since_collision {
            return 1.0;
        }

        if let Some(ticks_from_collision) = self.ticks_since_last_player_collision() {
            return if ticks_from_collision < DEFAULT_TICKS_TO_MAX_COMPRESSION {
                lerp(
                    1.0,
                    DEFAULT_MAX_COMPRESSION,
                    ticks_from_collision / DEFAULT_TICKS_TO_MAX_COMPRESSION,
                )
            } else if ticks_from_collision < DEFAULT_TICKS_TO_END_COMPRESSION {
                lerp(
                    DEFAULT_MAX_COMPRESSION,
                    1.0,
                    (ticks_from_collision - DEFAULT_TICKS_TO_MAX_COMPRESSION)
                        / (DEFAULT_TICKS_TO_END_COMPRESSION - DEFAULT_TICKS_TO_MAX_COMPRESSION),
                )
            } else {
                1.0
            };
        }
        1.0
    }

    fn player_is_in_boost(&self) -> bool {
        self.player.time_of_last_boost.is_some()
            && self.time_since(self.player.time_of_last_boost.unwrap())
                <= DEFAULT_MINIMUM_PARTICLE_GENERATION_TIME_AFTER_DASH
    }

    fn player_is_officially_fast(&self) -> bool {
        let mut inferred_speed = 0.0;
        if let Some(last_state) = self.player.recent_kinematic_states.get(0) {
            let last_pos = last_state.pos;
            inferred_speed = magnitude(self.player.pos - last_pos);
        }
        let actual_speed = magnitude(self.player.vel);
        actual_speed >= self.player.speed_of_blue || inferred_speed >= self.player.speed_of_blue
    }

    fn update_output_buffer(&mut self) {
        self.fill_output_buffer_with_black();
        self.draw_particles();
        self.draw_turrets();
        self.draw_non_air_blocks();

        if self.player.alive {
            self.draw_player();
        }
    }

    fn draw_player(&mut self) {
        let player_glyphs = self.get_player_glyphs();
        let grid_pos = snap_to_grid(self.player.pos);

        for i in 0..player_glyphs.len() {
            for j in 0..player_glyphs[i].len() {
                if let Some(glyph) = player_glyphs[i][j] {
                    let x = grid_pos.x() - 1 + i as i32;
                    let y = grid_pos.y() - 1 + j as i32;

                    if self.square_is_in_world(p(x, y)) {
                        self.output_buffer[x as usize][y as usize] = glyph;
                    }
                }
            }
        }
    }
    fn width(&self) -> usize {
        self.grid.len()
    }
    fn height(&self) -> usize {
        self.grid[0].len()
    }

    fn mid_square(&self) -> IPoint {
        p(self.width() as i32 / 2, self.height() as i32 / 2)
    }
    fn x_max(&self) -> i32 {
        self.width() as i32 - 1
    }
    fn y_max(&self) -> i32 {
        self.height() as i32 - 1
    }

    fn fill_output_buffer_with_black(&mut self) {
        let width = self.grid.len();
        let height = self.grid[0].len();
        for x in 0..width {
            for y in 0..height {
                self.output_buffer[x][y] = Glyph::from_char(' ');
            }
        }
    }

    fn draw_non_air_blocks(&mut self) {
        let width = self.grid.len();
        let height = self.grid[0].len();
        for x in 0..width {
            for y in 0..height {
                let block = self.grid[x][y];
                if block != Block::Air {
                    let mut glyph = Glyph::from_char(block.character());
                    glyph.fg_color = block.color();
                    self.output_buffer[x][y] = glyph;
                }
            }
        }
    }
    fn draw_turrets(&mut self) {
        //self.turrets.iter().for_each(|turret| {
        for i in 0..self.turrets.len() {
            let turret = &self.turrets[i];
            if let Some(laser_beam) = turret.laser_firing_result {
                self.draw_laser(laser_beam);
            }
        }
    }
    fn draw_laser(&mut self, laser_beam: SquarecastResult) {
        self.draw_visual_braille_line(
            laser_beam.start_pos,
            laser_beam.collider_pos,
            ColorName::Red,
        );
    }

    fn draw_particles(&mut self) {
        for i in 0..self.particles.len() {
            let particle_square = snap_to_grid(self.particles[i].pos);
            let mut grid_glyph = self.get_buffered_glyph(particle_square).clone();
            let particle_glyph = self.particles[i].glyph();
            if Glyph::is_braille(grid_glyph.character) {
                grid_glyph.character =
                    Glyph::add_braille(particle_glyph.character, grid_glyph.character);
            } else {
                grid_glyph = particle_glyph;
            }
            self.set_buffered_glyph(particle_square, grid_glyph);
        }
    }

    fn generate_speed_particles(&mut self) {
        let now = self.time_in_ticks();
        let then = now - self.get_time_factor();
        if let Some(&last_state) = self.player.recent_kinematic_states.get(0) {
            let last_pos = last_state.pos;
            match &self.player.speed_line_behavior {
                SpeedLineType::StillLine => {
                    self.place_static_speed_lines(last_pos, self.player.pos)
                }
                SpeedLineType::PerpendicularLines => self.place_perpendicular_moving_speed_lines(
                    last_state,
                    self.player.kinematic_state(),
                    then,
                    now,
                ),
                SpeedLineType::BurstChain => {
                    self.place_particle_burst_speed_line(last_pos, self.player.pos, 1.0)
                }
                _ => {}
            }
        }
    }
    fn place_particle_blast(&mut self, pos: Point<f32>, num_particles: i32, max_speed: f32) {
        for _i in 0..num_particles {
            let dir = random_direction();
            // want more even distribution over the blast
            let speed = rand_in_range(0.0f32, 1.0f32).sqrt() * max_speed;
            self.place_particle_with_velocity_and_lifetime(
                pos,
                dir * speed,
                self.player.speed_line_lifetime_in_ticks,
            );
        }
    }

    fn place_particle_burst(&mut self, pos: Point<f32>, num_particles: i32, speed: f32) {
        for i in 0..num_particles {
            let dir = radial(1.0, (i as f32 / num_particles as f32) * f32::TAU());
            self.place_particle_with_velocity_and_lifetime(
                pos,
                dir * speed,
                self.player.speed_line_lifetime_in_ticks,
            );
        }
    }

    fn place_grid_space_particle_burst(&mut self, pos: Point<f32>, num_particles: i32, speed: f32) {
        for i in 0..num_particles {
            let dir = radial(1.0, (i as f32 / num_particles as f32) * f32::TAU());
            let dir_in_grid_space = grid_space_to_world_space(dir, VERTICAL_GRID_STRETCH_FACTOR);
            self.place_particle_with_velocity_and_lifetime(
                pos,
                dir_in_grid_space * speed,
                self.player.speed_line_lifetime_in_ticks,
            );
        }
    }

    fn place_particle_burst_speed_line(
        &mut self,
        start_pos: Point<f32>,
        end_pos: Point<f32>,
        min_linear_density: f32,
    ) {
        let num_particles_per_burst = 64;
        let particle_speed = 0.05;
        for pos in points_in_line_with_max_gap(start_pos, end_pos, min_linear_density) {
            self.place_particle_burst(pos, num_particles_per_burst, particle_speed);
        }
    }

    fn place_perpendicular_moving_speed_lines(
        &mut self,
        start_state: KinematicState,
        end_state: KinematicState,
        start_time: f32,
        end_time: f32,
    ) {
        let particle_speed = DEFAULT_PARTICLE_SPEED;
        // Note: Due to non-square nature of the grid, player velocity may not be parallel to displacement
        let particles_per_block = 2.0;
        let time_frequency_of_speed_particles = particles_per_block * magnitude(self.player.vel);
        let time_period_of_speed_particles = 1.0 / time_frequency_of_speed_particles;
        for t in time_synchronized_interpolation_fractions(
            start_time,
            end_time,
            time_period_of_speed_particles,
        ) {
            // todo, better than lerp.  parabolic arcs for balistic trajectories
            let interpolated_kinematic_state = lerp(start_state, end_state, t);

            let pos = interpolated_kinematic_state.pos;
            let particle_vel = direction(interpolated_kinematic_state.vel) * particle_speed;
            self.place_particle_with_velocity_and_lifetime(
                pos,
                rotated_degrees(particle_vel, -90.0),
                self.player.speed_line_lifetime_in_ticks,
            );
            self.place_particle_with_velocity_and_lifetime(
                pos,
                rotated_degrees(particle_vel, 90.0),
                self.player.speed_line_lifetime_in_ticks,
            );
        }
    }

    fn place_static_speed_lines(&mut self, start: Point<f32>, end: Point<f32>) {
        // draw all corners
        let mut offsets = Vec::<Point<f32>>::new();
        let n = 1;
        let r = 0.3;
        for i in -n..=n {
            for j in -n..=n {
                if i == 0 || j == 0 {
                    continue;
                }
                let x = i as f32 * r / n as f32;
                let y = j as f32 * r / n as f32;
                offsets.push(p(x, y));
            }
        }
        for offset in offsets {
            self.place_line_of_particles(start + offset, end + offset);
        }
    }

    fn get_player_glyphs(&self) -> Vec<Vec<Option<Glyph>>> {
        if self.get_player_compression_fraction() == 1.0 {
            Glyph::get_glyphs_for_colored_floating_square(self.player.pos, self.get_player_color())
        } else {
            vec![
                vec![None, None, None],
                vec![None, Some(self.get_compressed_player_glyph()), None],
                vec![None, None, None],
            ]
        }
    }

    fn get_compressed_player_glyph(&self) -> Glyph {
        let player_compression = self.get_player_compression_fraction();
        let collision_normal = self.player.last_collision.as_ref().unwrap().normal;
        if collision_normal.x() != 0 {
            Glyph::colored_square_with_horizontal_offset(
                -collision_normal.x().sign() as f32 * (1.0 - player_compression),
                self.get_player_color(),
            )
        } else if collision_normal.y() != 0 {
            Glyph::colored_square_with_vertical_offset(
                -collision_normal.y().sign() as f32 * (1.0 - player_compression),
                self.get_player_color(),
            )
        } else {
            panic!("There was a collision with no normal.  What?");
        }
    }

    fn get_occupancy_of_nearby_walls(&self, square: Point<i32>) -> LocalBlockOccupancy {
        let mut output = empty_local_block_occupancy();
        for i in 0..3 {
            for j in 0..3 {
                let rx = i as i32 - 1;
                let ry = j as i32 - 1;
                output[i][j] = if let Some(block) = self.try_get_block(square + p(rx, ry)) {
                    block != Block::Air
                } else {
                    false
                };
            }
        }
        output
    }

    fn get_buffered_glyph(&self, pos: Point<i32>) -> &Glyph {
        return &self.output_buffer[pos.x() as usize][pos.y() as usize];
    }
    fn set_buffered_glyph(&mut self, pos: Point<i32>, new_glyph: Glyph) {
        self.output_buffer[pos.x() as usize][pos.y() as usize] = new_glyph;
    }
    #[allow(dead_code)]
    fn get_glyph_on_screen(&self, pos: Point<i32>) -> &Glyph {
        return &self.output_on_screen[pos.x() as usize][pos.y() as usize];
    }

    fn print_output_buffer(&self) {
        for y in 0..self.height() {
            let reverse_y = self.height() - 1 - y;
            let mut row_string = String::new();
            for x in 0..self.width() {
                row_string += &self.output_buffer[x][reverse_y].to_string();
            }
            row_string += &Glyph::reset_colors();
            if reverse_y % 5 == 0 {
                row_string += &format!("-- {}", reverse_y);
            }
            println!("{}", row_string);
        }
    }

    fn update_screen(&mut self, stdout: &mut MouseTerminal<RawTerminal<std::io::Stdout>>) {
        let width = self.grid.len();
        let height = self.grid[0].len();
        // Now update the graphics where applicable
        for x in 0..width {
            for y in 0..height {
                if self.output_buffer[x][y] != self.output_on_screen[x][y] {
                    let (term_x, term_y) = self.world_to_screen(&(x as i32, y as i32));
                    write!(stdout, "{}", termion::cursor::Goto(term_x, term_y)).unwrap();
                    write!(stdout, "{}", self.output_buffer[x][y].to_string()).unwrap();
                }
            }
        }
        write!(
            stdout,
            "{}{}",
            termion::cursor::Goto(1, 1),
            self.fps() as i32
        )
        .unwrap();
        stdout.flush().unwrap();
        self.output_on_screen = self.output_buffer.clone();
    }

    fn fps(&self) -> f32 {
        if self.recent_tick_durations_s.is_empty() {
            return 0.0;
        }
        let avg_s_per_frame = self.recent_tick_durations_s.iter().sum::<f32>()
            / self.recent_tick_durations_s.len() as f32;
        let fps = 1.0 / avg_s_per_frame;
        if fps > 100.0 {
            0.0
        } else {
            fps
        }
    }

    fn apply_gravity_to_blocks(&mut self) {
        // We want to count from bottom to top, because things fall down
        for x in 0..self.terminal_size.0 as usize {
            for y in 0..self.terminal_size.1 as usize {
                let block = self.grid[x][y];
                if block.subject_to_block_gravity() {
                    let is_bottom_row = y == 0;
                    let has_direct_support = !is_bottom_row && self.grid[x][y - 1] != Block::Air;
                    if is_bottom_row {
                        self.grid[x][y] = Block::Air;
                    } else if !has_direct_support {
                        self.grid[x][y - 1] = block;
                        self.grid[x][y] = Block::Air;
                    }
                }
            }
        }
    }

    fn kill_player(&mut self) {
        self.player.alive = false;
        self.place_particle_blast(self.player.pos, 100, 3.0);
    }

    fn player_is_grabbing_wall(&self) -> bool {
        self.player_is_pressing_against_wall_horizontally()
            && self.player.vel.y() <= 0.0
            && !self.player_is_standing_on_block()
    }

    fn player_is_running_up_wall(&self) -> bool {
        self.player_is_touching_wall_horizontally() && self.player.vel.y() > 0.0
    }
    fn player_is_touching_wall_horizontally(&self) -> bool {
        self.player_exactly_touching_wall_in_direction(right_i())
            || self.player_exactly_touching_wall_in_direction(left_i())
    }

    fn get_lone_touched_wall_direction(&self) -> Option<IPoint> {
        let touching_left = self.player_exactly_touching_wall_in_direction(left_i());
        let touching_right = self.player_exactly_touching_wall_in_direction(right_i());
        if touching_left == touching_right {
            None
        } else if touching_left {
            Some(left_i())
        } else {
            Some(right_i())
        }
    }

    fn player_is_pressing_against_wall_horizontally(&self) -> bool {
        if self.player.desired_direction.x() == 0 {
            return false;
        }
        let horizontal_desired_direction = p(self.player.desired_direction.x(), 0);
        return self.player_exactly_touching_wall_in_direction(horizontal_desired_direction);
    }

    fn apply_player_wall_friction(&mut self, dt_in_ticks: f32) {
        self.player.vel.set_y(decelerate_linearly_to_cap(
            self.player.vel.y(),
            0.0,
            self.player.acceleration_from_floor_traction * dt_in_ticks,
        ));
    }

    fn apply_player_floor_traction(&mut self, dt_in_ticks: f32) {
        if self.player_is_pressing_against_wall_horizontally() {
            return;
        }
        self.player.vel.set_x(accelerate_within_max_speed(
            self.player.vel.x(),
            self.player.desired_direction.x(),
            self.player.max_run_speed,
            self.player.acceleration_from_floor_traction * dt_in_ticks,
        ));
    }

    fn apply_player_air_traction(&mut self, dt_in_ticks: f32) {
        if self.player_is_pressing_against_wall_horizontally() {
            return;
        }
        self.player.vel.set_x(accelerate_within_max_speed(
            self.player.vel.x(),
            self.player.desired_direction.x(),
            self.player.max_midair_move_speed,
            self.player.acceleration_from_air_traction * dt_in_ticks,
        ));
    }

    fn hangtime_speed_threshold(&self) -> f32 {
        self.player.jump_properties.delta_vy * DEFAULT_HANGTIME_JUMP_SPEED_FRACTION_THRESHOLD
    }

    fn get_gravitational_acceleration(&mut self) -> f32 {
        let gravity_multiplier = if self.player.vel.y().abs() < self.hangtime_speed_threshold() {
            DEFAULT_HANGTIME_GRAVITY_MULTIPLIER
        } else {
            1.0
        };
        return self.player.jump_properties.g * gravity_multiplier;
    }

    fn apply_player_kinematics(&mut self, dt_in_ticks: f32) {
        let start_kinematic_state = KinematicState {
            pos: self.player.pos,
            vel: self.player.vel,
            accel: self.player.accel,
        };

        let mut remaining_time = dt_in_ticks;
        let mut collision_occurred = false;
        let mut collisions_that_happened = Vec::new();

        let mut timeout_counter = 0;
        while remaining_time > 0.0 {
            timeout_counter += 1;
            if timeout_counter > 100 {
                dbg!(start_kinematic_state);
                dbg!(remaining_time);
                dbg!(collisions_that_happened.last());
                panic!("player kinematics looped too much :( ");
            }

            let kinematic_state_at_start_of_submove = KinematicState {
                pos: self.player.pos,
                vel: self.player.vel,
                accel: self.player.accel,
            };

            let mut target_state_delta = if self.player_wants_to_come_to_a_full_stop() {
                kinematic_state_at_start_of_submove
                    .extrapolated_delta_with_full_stop_at_slowest(remaining_time)
            } else if self.player_is_supported() {
                kinematic_state_at_start_of_submove
                    .extrapolated_delta_with_speed_cap(remaining_time, self.player.max_run_speed)
            } else {
                kinematic_state_at_start_of_submove.extrapolated_delta(remaining_time)
            };
            target_state_delta.pos =
                world_space_to_grid_space(target_state_delta.pos, VERTICAL_GRID_STRETCH_FACTOR);
            let target_state: KinematicState =
                kinematic_state_at_start_of_submove + target_state_delta;

            let maybe_collision = self.unit_squarecast(self.player.pos, target_state.pos);

            let mut end_state = target_state;

            if maybe_collision.hit_something() {
                let collision = maybe_collision;
                collision_occurred = true;
                collisions_that_happened.push(collision.clone());

                let fraction_through_remaining_movement_just_moved = inverse_lerp_2d(
                    start_kinematic_state.pos,
                    target_state.pos,
                    collision.collider_pos,
                );
                let rel_collision_time =
                    remaining_time * fraction_through_remaining_movement_just_moved;

                remaining_time -= rel_collision_time;

                end_state.pos = collision.collider_pos;

                let extrapolated_to_collision_time = if self.player_wants_to_come_to_a_full_stop() {
                    kinematic_state_at_start_of_submove
                        .extrapolated_with_full_stop_at_slowest(rel_collision_time)
                } else if self.player_is_supported() {
                    kinematic_state_at_start_of_submove
                        .extrapolated_with_speed_cap(rel_collision_time, self.player.max_run_speed)
                } else {
                    kinematic_state_at_start_of_submove.extrapolated(rel_collision_time)
                };
                // Not pos, because we trust the collision check point more than the extrapolation point for exactness
                end_state.vel = extrapolated_to_collision_time.vel;
                end_state.accel = extrapolated_to_collision_time.accel;

                self.player.last_collision = Some(PlayerBlockCollision {
                    time_in_ticks: self.time_in_ticks() - remaining_time,
                    normal: collision.collision_normal.unwrap(),
                    collider_velocity: end_state.vel,
                    collider_pos: collision.collider_pos,
                    collided_block_square: collision.collided_block_square.unwrap(),
                });

                if self.internal_corner_behavior == InternalCornerBehavior::RedirectPlayer
                    && self.pos_is_exactly_on_square(end_state.pos)
                    && self.pos_is_in_deep_corner(end_state.pos)
                {
                    let player_square_adjacency = self.try_get_player_square_adjacency().unwrap();
                    let symmetry_axis: FPoint = inside_direction_of_corner(player_square_adjacency);
                    end_state.vel = -reflect_vector_over_axis(end_state.vel, symmetry_axis);
                    // not super sure about this one
                    end_state.accel = -reflect_vector_over_axis(end_state.accel, symmetry_axis);
                }
                if collision.collision_normal.unwrap().x() == 0 {
                    end_state.vel.set_y(0.0);
                    end_state.accel.set_y(0.0);
                } else if collision.collision_normal.unwrap().y() == 0 {
                    end_state.vel.set_x(0.0);
                    end_state.accel.set_x(0.0);
                } else {
                    panic!("bad collision normal: {:?}", collision);
                }

                // collision from acceleration this tick
                if kinematic_state_at_start_of_submove == target_state {
                    dbg!(
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
                        start_kinematic_state,
                        kinematic_state_at_start_of_submove,
                    );
                    panic!("No kinematic state change from collision");
                }
                //dbg!("after collision", target_state, remaining_time);
            } else {
                // should exit loop
                remaining_time = 0.0;
            }

            // set the player to the current target
            self.player.pos = end_state.pos;
            self.player.vel = end_state.vel;
            self.player.accel = end_state.accel;

            if !self.square_is_in_world(snap_to_grid(self.player.pos)) {
                // Player went out of bounds and died
                self.kill_player();
                return;
            }
            if remaining_time < 0.00001 {
                remaining_time = 0.0;
            }
        }

        let step_taken = self.player.pos - start_kinematic_state.pos;
        self.save_recent_player_kinematic_state(start_kinematic_state);

        if collision_occurred {
            self.player.moved_normal_to_collision_since_collision = false;
        } else {
            if self.movement_is_normal_to_last_collision(step_taken) {
                self.player.moved_normal_to_collision_since_collision = true;
            }
        }
    }

    fn deflect_off_collision_plane(
        &mut self,
        collision: SquarecastResult,
        move_start: FPoint,
        relative_target: FPoint,
        vel: FPoint,
    ) -> (FPoint, FPoint, FPoint) {
        let step_taken_to_this_collision = collision.collider_pos - move_start;

        let mut new_relative_target = relative_target - step_taken_to_this_collision;
        let mut new_vel = vel;
        if collision.collision_normal.unwrap().x() != 0 {
            new_vel.set_x(0.0);
            new_relative_target = project_a_onto_b(relative_target, up_f());
        } else if collision.collision_normal.unwrap().y() != 0 {
            new_vel.set_y(0.0);
            new_relative_target = project_a_onto_b(relative_target, right_f());
        } else {
            panic!("collision has zero normal");
        }
        let new_move_start = collision.collider_pos;
        (new_move_start, new_relative_target, new_vel)
    }

    fn player_square(&self) -> IPoint {
        snap_to_grid(self.player.pos)
    }

    fn movement_is_normal_to_last_collision(&mut self, step_taken: FPoint) -> bool {
        self.player.last_collision.is_some()
            && step_taken.dot(floatify(
                self.player.last_collision.as_ref().unwrap().normal,
            )) != 0.0
    }

    fn save_recent_player_kinematic_state(&mut self, state: KinematicState) {
        self.player.recent_kinematic_states.push_front(state);
        while self.player.recent_kinematic_states.len() > NUM_SAVED_PLAYER_KINEMATIC_STATES as usize
        {
            self.player.recent_kinematic_states.pop_back();
        }
    }

    fn get_square_adjacency(&self, square: IPoint) -> Option<LocalBlockOccupancy> {
        let mut local_block_occupancy = empty_local_block_occupancy();
        for rel_square in get_3x3_squares() {
            let abs_square = rel_square + square;
            let abs_square_blocks_player = if let Some(block) = self.try_get_block(abs_square) {
                block.can_collide_with_player()
            } else {
                return None;
            };
            let index_2d = rel_square + p(1, 1);
            local_block_occupancy[index_2d.x() as usize][index_2d.y() as usize] =
                abs_square_blocks_player;
        }
        Some(local_block_occupancy)
    }

    fn player_is_sliding_down_wall(&self) -> bool {
        self.player_is_grabbing_wall() && self.player.vel.y() <= 0.0
    }

    fn update_player_accelerations(&mut self) {
        let DEBUG_PRINT_ACCELERATION = false;

        let mut total_acceleration = zero_f();

        if DEBUG_PRINT_ACCELERATION {
            dbg!(total_acceleration);
        }
        let traction_acceleration_magnitude = if self.player_is_supported() {
            self.player.acceleration_from_floor_traction
        } else {
            self.player.acceleration_from_air_traction
        };
        if self.player_is_grabbing_wall() {
            // wall friction
            if self.player_is_sliding_down_wall() {
                if DEBUG_PRINT_ACCELERATION {
                    dbg!(total_acceleration);
                }
                total_acceleration
                    .add_assign(up_f() * self.player.acceleration_from_floor_traction);
                if DEBUG_PRINT_ACCELERATION {
                    dbg!(total_acceleration);
                }
            }
        } else {
            if DEBUG_PRINT_ACCELERATION {
                dbg!(total_acceleration);
            }
            // floor/air traction.  Active horizontal motion
            let player_can_faster_x = self.player.vel.x().abs() < self.player.max_run_speed;
            let player_want_faster_x =
                (self.player.desired_direction.x() as f32 * self.player.vel.x() > 0.0)
                    || (self.player.desired_direction.x() != 0 && self.player.vel.x() == 0.0);
            let player_want_slower_x =
                self.player.desired_direction.x() as f32 * self.player.vel.x() < 0.0;
            let player_want_stop_x = self.player.desired_direction.x() == 0;
            // Apply traction
            if (player_can_faster_x && player_want_faster_x) || player_want_slower_x {
                if DEBUG_PRINT_ACCELERATION {
                    dbg!(total_acceleration);
                }
                total_acceleration.add_assign(
                    project_a_onto_b(floatify(self.player.desired_direction), right_f())
                        * traction_acceleration_magnitude,
                );
                if DEBUG_PRINT_ACCELERATION {
                    dbg!(total_acceleration);
                }
            } else if player_want_stop_x {
                if DEBUG_PRINT_ACCELERATION {
                    dbg!(total_acceleration);
                }
                total_acceleration.add_assign(
                    project_a_onto_b(direction(-self.player.vel), right_f())
                        * traction_acceleration_magnitude,
                );
                if total_acceleration.x().is_nan() {
                    dbg!(
                        self.player.vel,
                        direction(self.player.vel),
                        project_a_onto_b(direction(-self.player.vel), right_f())
                    );
                }
                if DEBUG_PRINT_ACCELERATION {
                    dbg!(total_acceleration);
                }
            }
            if DEBUG_PRINT_ACCELERATION {
                dbg!(total_acceleration);
            }
            let moving_up = self.player.vel.y() > 0.0;
            let on_ground = self.player_is_supported() && !moving_up;
            if on_ground {
                if DEBUG_PRINT_ACCELERATION {
                    dbg!(total_acceleration);
                }
                // floor friction
                let fast_enough_for_friction_x =
                    self.player.vel.x().abs() > self.player.ground_friction_start_speed;
                if fast_enough_for_friction_x || !player_want_faster_x {
                    total_acceleration.add_assign(
                        direction(-self.player.vel) * self.player.deceleration_from_ground_friction,
                    );
                }
                if DEBUG_PRINT_ACCELERATION {
                    dbg!(total_acceleration);
                }
            } else {
                if DEBUG_PRINT_ACCELERATION {
                    dbg!(total_acceleration);
                }
                // gravity
                total_acceleration.add_assign(down_f() * self.get_gravitational_acceleration());

                if DEBUG_PRINT_ACCELERATION {
                    dbg!(total_acceleration);
                }
                // air friction
                if magnitude(self.player.vel) > self.player.air_friction_start_speed {
                    total_acceleration.add_assign(
                        direction(-self.player.vel) * self.player.deceleration_from_air_friction,
                    );
                }
                if DEBUG_PRINT_ACCELERATION {
                    dbg!(total_acceleration);
                }
            }
        }
        if DEBUG_PRINT_ACCELERATION {
            dbg!(total_acceleration);
        }

        self.player.accel = total_acceleration;
    }

    fn apply_player_air_friction(&mut self, dt_in_ticks: f32) {
        self.player.vel.set_x(decelerate_linearly_to_cap(
            self.player.vel.x(),
            self.player.air_friction_start_speed,
            self.player.deceleration_from_air_friction * dt_in_ticks,
        ));

        self.player.vel.set_y(decelerate_linearly_to_cap(
            self.player.vel.y(),
            self.player.air_friction_start_speed,
            self.player.deceleration_from_air_friction * dt_in_ticks,
        ));
    }

    fn apply_player_floor_friction(&mut self, dt_in_ticks: f32) {
        self.player.vel.set_x(decelerate_linearly_to_cap(
            self.player.vel.x(),
            self.player.ground_friction_start_speed,
            self.player.deceleration_from_ground_friction * dt_in_ticks,
        ));
    }

    // Where the player can move to in a line
    // tries to draw a line in air
    // returns None if out of bounds
    // returns the start position if start is not Block::Air
    fn unit_squarecast(&self, start_pos: Point<f32>, end_pos: Point<f32>) -> SquarecastResult {
        self.squarecast_for_player_collision(start_pos, end_pos, 1.0)
    }

    fn linecast_laser(&self, start_pos: Point<f32>, end_pos: Point<f32>) -> SquarecastResult {
        let filter = Box::new(|block: Block| block.can_be_hit_by_laser());
        first_hit(vec![
            self.linecast_with_block_filter(start_pos, end_pos, filter),
            self.linecast_particles_only(start_pos, end_pos),
        ])
    }
    fn linecast_walls_only(&self, start_pos: Point<f32>, end_pos: Point<f32>) -> SquarecastResult {
        self.squarecast_one_block_type(start_pos, end_pos, 0.0, Block::Wall)
    }
    fn linecast_particles_only(&self, start_pos: FPoint, end_pos: FPoint) -> SquarecastResult {
        let LINECAST_PARTICLE_DETECTION_RADIUS = 0.2;
        // todo: maybe precalculate the location map?
        self.squarecast_particles_only(
            start_pos,
            end_pos,
            LINECAST_PARTICLE_DETECTION_RADIUS * 2.0,
            Some(self.get_particle_location_map()),
        )
    }

    fn squarecast_one_block_type(
        &self,
        start_pos: Point<f32>,
        end_pos: Point<f32>,
        moving_square_side_length: f32,
        hittable_block: Block,
    ) -> SquarecastResult {
        let filter = Box::new(move |block| block == hittable_block);
        self.squarecast_with_block_filter(start_pos, end_pos, moving_square_side_length, filter)
    }

    fn linecast_for_non_air_blocks(
        &self,
        start_pos: Point<f32>,
        end_pos: Point<f32>,
    ) -> SquarecastResult {
        let filter = Box::new(|block| block != Block::Air);
        self.squarecast_with_block_filter(start_pos, end_pos, 0.0, filter)
    }

    fn squarecast_for_non_air_blocks(
        &self,
        start_pos: Point<f32>,
        end_pos: Point<f32>,
        moving_square_side_length: f32,
    ) -> SquarecastResult {
        let filter = Box::new(|block| block != Block::Air);
        self.squarecast_with_block_filter(start_pos, end_pos, moving_square_side_length, filter)
    }

    fn squarecast_particles_only(
        &self,
        start_pos: Point<f32>,
        end_pos: Point<f32>,
        moving_square_side_length: f32,
        particle_location_map: Option<&ParticleLocationMap>,
    ) -> SquarecastResult {
        let falsefilter = Box::new(|_| false);
        self.squarecast(
            start_pos,
            end_pos,
            moving_square_side_length,
            falsefilter,
            particle_location_map,
        )
    }

    fn squarecast_for_player_collision(
        &self,
        start_pos: Point<f32>,
        end_pos: Point<f32>,
        moving_square_side_length: f32,
    ) -> SquarecastResult {
        self.squarecast_with_block_filter(
            start_pos,
            end_pos,
            moving_square_side_length,
            Box::new(|block| block.can_collide_with_player()),
        )
    }

    fn linecast_with_block_filter(
        &self,
        start_pos: Point<f32>,
        end_pos: Point<f32>,
        block_filter: BlockFilter,
    ) -> SquarecastResult {
        self.squarecast_with_block_filter(start_pos, end_pos, 0.0, block_filter)
    }

    fn squarecast_with_block_filter(
        &self,
        start_pos: Point<f32>,
        end_pos: Point<f32>,
        moving_square_side_length: f32,
        block_filter: BlockFilter,
    ) -> SquarecastResult {
        self.squarecast(
            start_pos,
            end_pos,
            moving_square_side_length,
            block_filter,
            None,
        )
    }

    fn squarecast(
        &self,
        start_pos: Point<f32>,
        end_pos: Point<f32>,
        moving_square_side_length: f32,
        block_filter: BlockFilter,
        particle_location_map: Option<&ParticleLocationMap>,
    ) -> SquarecastResult {
        let default_result = SquarecastResult::no_hit_result(start_pos, end_pos);
        let mut intermediate_player_positions_to_check = lin_space_from_start_2d(
            start_pos,
            end_pos,
            self.num_positions_per_block_to_check_for_collisions,
        );
        // needed for very small steps
        intermediate_player_positions_to_check.push(end_pos);
        for point_to_check in &intermediate_player_positions_to_check {
            let overlapping_squares = grid_squares_touched_or_overlapped_by_floating_square(
                *point_to_check,
                moving_square_side_length,
            );
            let mut collisions = Vec::<SquarecastResult>::new();
            for overlapping_square in &overlapping_squares {
                if let Some(unwrapped_particle_map) = particle_location_map.as_ref() {
                    if unwrapped_particle_map.contains_key(overlapping_square) {
                        for particle_index in
                            unwrapped_particle_map.get(overlapping_square).unwrap()
                        {
                            if self.particles[*particle_index]
                                .in_floating_square(*point_to_check, moving_square_side_length)
                            {
                                collisions.push(SquarecastResult {
                                    start_pos,
                                    unrounded_collider_pos: *point_to_check,
                                    collider_pos: *point_to_check,
                                    collision_normal: None,
                                    collided_block_square: None,
                                    collided_particle_index: Some(*particle_index),
                                });
                            }
                        }
                    }
                }
                if let Some(block) = self.try_get_block(*overlapping_square) {
                    if block.can_collide_with_player() && block_filter(block) {
                        let adjacent_occupancy =
                            self.get_occupancy_of_nearby_walls(*overlapping_square);
                        let collision = single_block_squarecast_with_edge_extensions(
                            start_pos,
                            end_pos,
                            *overlapping_square,
                            moving_square_side_length,
                            adjacent_occupancy,
                        );
                        if collision.hit_something() {
                            collisions.push(collision);
                        }
                    }
                }
            }
            if !collisions.is_empty() {
                let closest_collision_to_start = first_hit(collisions);
                //dbg!(&closest_collision_to_start);

                // might have missed one
                if closest_collision_to_start.hit_block() {
                    let normal_square = closest_collision_to_start.collided_block_square.unwrap()
                        + closest_collision_to_start.collision_normal.unwrap();
                    // TODO: have this account for block expansion from other adjacent blocks?
                    if !point_inside_grid_square(start_pos, normal_square)
                        && self.square_is_in_world(normal_square)
                        && self.get_block(normal_square).can_collide_with_player()
                        && block_filter(self.get_block(normal_square))
                    {
                        let adjacent_occupancy = self.get_occupancy_of_nearby_walls(normal_square);
                        let collision = single_block_squarecast_with_edge_extensions(
                            start_pos,
                            end_pos,
                            normal_square,
                            moving_square_side_length,
                            adjacent_occupancy,
                        );
                        if collision.hit_something() {
                            //dbg!( start_pos, end_pos, normal_square, moving_square_side_length, adjacent_occupancy, &collision );
                            return collision;
                        } else {
                            // Need to disable this panic due to the case of internal corners.  the corner block expands upwards from having a block orthogonal, and the player may collide with that corner block when moving into a corner as a result.  The block normal to that collision is the block the player is standing on.
                            //panic!("No collision with wall block normal to collision");
                        }
                    }
                }
                return closest_collision_to_start;
            }
        }
        return default_result;
    }

    fn player_is_supported(&self) -> bool {
        return self.player_is_standing_on_block();
    }

    fn player_is_standing_on_block(&self) -> bool {
        self.player_exactly_touching_wall_in_direction(down_i())
    }

    fn get_block_relative_to_player(&self, rel_pos: Point<i32>) -> Option<Block> {
        let target_pos = snap_to_grid(self.player.pos) + rel_pos;
        if self.player.alive && self.square_is_in_world(target_pos) {
            return Some(self.get_block(target_pos));
        }
        return None;
    }
    fn player_exactly_touching_wall_in_direction(&self, direction: Point<i32>) -> bool {
        let direction_f = floatify(direction);
        let collision = self.unit_squarecast(self.player.pos, self.player.pos + direction_f * 0.1);
        if collision.hit_something() {
            project_a_onto_b(collision.collider_pos, direction_f)
                == project_a_onto_b(self.player.pos, direction_f)
        } else {
            false
        }
    }

    fn square_is_in_world(&self, pos: Point<i32>) -> bool {
        pos.x() >= 0
            && pos.x() < self.terminal_size.0 as i32
            && pos.y() >= 0
            && pos.y() < self.terminal_size.1 as i32
    }
    fn square_is_empty(&mut self, square: IPoint) -> bool {
        self.square_is_in_world(square) && matches!(self.get_block(square), Block::Air)
    }

    fn player_wants_to_come_to_a_full_stop(&self) -> bool {
        self.player_is_grabbing_wall()
            || (self.player_is_supported() && self.player.desired_direction.x() == 0)
    }
    fn player_is_before_peak_of_jump(&self) -> bool {
        !self.player_is_supported() && self.player.vel.y() > 0.0
    }
}
fn init_platformer_test_world(width: u16, height: u16) -> Game {
    let mut game = Game::new(width, height);
    game.place_line_of_blocks((2, 3), (8, 3), Block::Wall);
    game.place_player(5.0, 5.0);

    let num_platforms = 5;
    let platform_width = 28;
    let horizontal_buffer = 3;
    let vertical_buffer = horizontal_buffer;
    for i in 0..num_platforms {
        let x_start = rand_in_range(
            horizontal_buffer,
            game.x_max() - (platform_width + horizontal_buffer),
        );
        let height = rand_in_range(vertical_buffer, game.y_max() - vertical_buffer);
        game.place_line_of_blocks(
            (x_start, height),
            (x_start + platform_width, height),
            Block::Wall,
        );
    }

    let num_turrets = 2;
    for i in 0..num_turrets {
        let x = rand_in_range(horizontal_buffer, game.x_max() - horizontal_buffer);
        let y = rand_in_range(vertical_buffer, game.y_max() - vertical_buffer);
        game.place_turret(p(x, y));
    }

    game
}
fn init_test_world_1(width: u16, height: u16) -> Game {
    let mut game = Game::new(width, height);
    //let mut game = Game::new(10, 40);
    //game.set_player_jump_delta_v(1.0);
    //game.player.jump_properties.g = 0.05;
    //game.player.acceleration_from_floor_traction = 0.6;
    //game.set_player_max_run_speed(0.7);

    game.place_boundary_wall();

    let bottom_left = (
        (game.terminal_size.0 * 2 / 5) as i32,
        (game.terminal_size.1 / 4) as i32,
    );
    game.place_line_of_blocks(
        bottom_left,
        ((4 * game.terminal_size.0 / 5) as i32, bottom_left.1),
        Block::Wall,
    );
    game.place_line_of_blocks(
        bottom_left,
        (bottom_left.0, 3 * (game.terminal_size.1 / 4) as i32),
        Block::Wall,
    );

    game.place_wall_rect(
        p(game.width() as i32 / 6, game.height() as i32 / 6),
        p(game.width() as i32 / 5, game.height() as i32 / 5),
    );
    let cube_start = p(2, 2);
    for x in 0..4 {
        for y in 0..4 {
            game.place_turret(cube_start + p(x, y));
        }
    }

    game.place_step_foe(p(game.width() as i32 - 3, 1));

    let start_square = p(1, game.height() as i32 - 2);
    let step = right_i();
    for i in 1..DEFAULT_PARTICLES_IN_AMALGAMATION_FOR_EXPLOSION {
        game.place_block(start_square + step * i, Block::ParticleAmalgam(i));
    }

    game.place_player(
        game.terminal_size.0 as f32 / 2.0,
        game.terminal_size.1 as f32 / 2.0,
    );
    game.player.vel.set_x(0.1);
    game
}

fn seconds_to_ticks(s: f32) -> f32 {
    s * MAX_FPS
}

fn set_up_panic_hook() {
    std::panic::set_hook(Box::new(move |panic_info| {
        write!(stdout(), "{}", termion::screen::ToMainScreen);
        write!(stdout(), "{:?}", panic_info);
    }));
}

fn set_up_input_thread() -> Receiver<Event> {
    let (tx, rx) = channel();
    thread::spawn(move || {
        for c in stdin().events() {
            let evt = c.unwrap();
            tx.send(evt).unwrap();
        }
    });
    rx
}

fn main() {
    let (width, height) = termion::terminal_size().unwrap();
    let mut game = init_test_world_1(width, height);
    //let mut game = init_platformer_test_world(width, height);

    let mut terminal = termion::screen::AlternateScreen::from(termion::cursor::HideCursor::from(
        MouseTerminal::from(stdout().into_raw_mode().unwrap()),
    ));

    set_up_panic_hook();

    // Separate thread for reading input
    let event_receiver = set_up_input_thread();

    let mut prev_start_time = Instant::now();
    while game.running {
        let start_time = Instant::now();
        let prev_tick_duration_ms = start_time.duration_since(prev_start_time).as_millis();
        let prev_tick_duration_s: f32 = prev_tick_duration_ms as f32 / 1000.0;
        prev_start_time = start_time;

        game.recent_tick_durations_s
            .push_front(prev_tick_duration_s);
        if game.recent_tick_durations_s.len() > 10 {
            game.recent_tick_durations_s.pop_back();
        }

        while let Ok(event) = event_receiver.try_recv() {
            game.handle_event(event);
        }
        game.tick_physics();
        game.update_output_buffer();
        game.update_screen(&mut terminal);
        let tick_duration_so_far_ms = start_time.elapsed().as_millis();
        if tick_duration_so_far_ms < IDEAL_FRAME_DURATION_MS {
            thread::sleep(Duration::from_millis(
                (IDEAL_FRAME_DURATION_MS - tick_duration_so_far_ms) as u64,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::assert;
    use std::collections::HashSet;
    use std::iter::Map;

    fn set_up_game() -> Game {
        Game::new(30, 30)
    }

    fn print_grid(game: &mut Game) {
        game.update_output_buffer();
        game.print_output_buffer();
    }

    fn set_up_tall_drop() -> Game {
        let mut game = Game::new(20, 40);
        game.player.jump_properties.g = 0.05;
        game.player.acceleration_from_floor_traction = 0.6;
        game.set_player_max_run_speed(0.7);

        game.place_boundary_wall();

        let bottom_left = (
            (game.terminal_size.0 * 2 / 5) as i32,
            (game.terminal_size.1 / 4) as i32,
        );
        game.place_line_of_blocks(
            bottom_left,
            ((4 * game.terminal_size.0 / 5) as i32, bottom_left.1),
            Block::Wall,
        );
        game.place_line_of_blocks(
            bottom_left,
            (bottom_left.0, 3 * (game.terminal_size.1 / 4) as i32),
            Block::Wall,
        );
        game.place_player(
            game.terminal_size.0 as f32 / 2.0,
            game.terminal_size.1 as f32 / 2.0,
        );
        game
    }

    fn set_up_game_filled_with_walls() -> Game {
        let mut game = Game::new(30, 30);
        for y in 0..game.height() {
            game.place_line_of_blocks(
                (0, y as i32),
                (game.width() as i32 - 1, y as i32),
                Block::Wall,
            );
        }
        game
    }

    fn set_up_player_in_bottom_right_wall_corner() -> Game {
        let mut game = Game::new(30, 30);
        game.place_boundary_wall();
        game.place_player(1.0, game.width() as f32 - 2.0);
        game
    }

    fn set_up_just_player() -> Game {
        let mut game = set_up_game();
        game.place_player(15.0, 11.0);
        return game;
    }

    fn set_up_player_in_zero_g() -> Game {
        let mut game = set_up_just_player();
        game.player.jump_properties.g = 0.0;
        return game;
    }

    fn set_up_player_just_dashed_right_in_zero_g() -> Game {
        let mut game = set_up_player_in_zero_g();
        game.player.desired_direction = p(1, 0);
        game.player_dash();
        game
    }

    fn set_up_player_barely_fighting_air_friction_to_the_right_in_zero_g() -> Game {
        let mut game = set_up_player_in_zero_g();
        game.player.desired_direction = right_i();
        game.player.vel = right_f()
            * (game.player.air_friction_start_speed
                + game.player.deceleration_from_air_friction * 0.9);
        game
    }

    fn set_up_player_barely_fighting_air_friction_up_in_zero_g() -> Game {
        let mut game = set_up_player_in_zero_g();
        game.player.desired_direction = p(0, 1);
        game.player.vel = up_f()
            * (game.player.air_friction_start_speed
                + game.player.deceleration_from_air_friction * 0.9);
        game
    }

    fn set_up_player_in_zero_g_frictionless_vacuum() -> Game {
        let mut game = set_up_player_in_zero_g();
        be_frictionless(&mut game);
        be_in_vacuum(&mut game);
        game
    }

    fn set_up_player_on_platform() -> Game {
        let mut game = set_up_just_player();
        let platform_y = game.player.pos.y() as i32 - 1;
        game.place_line_of_blocks((0, platform_y), (29, platform_y), Block::Wall);
        return game;
    }

    fn set_up_player_on_amalgam_platform() -> Game {
        let mut game = set_up_just_player();
        let platform_y = game.player.pos.y() as i32 - 1;
        game.place_line_of_blocks(
            (10, platform_y),
            (20, platform_y),
            Block::ParticleAmalgam(DEFAULT_PARTICLE_DENSITY_FOR_AMALGAMATION),
        );
        return game;
    }

    fn get_vertical_jump_landing_player_block_collision(player: &Player) -> PlayerBlockCollision {
        PlayerBlockCollision {
            time_in_ticks: 0.0,
            normal: p(0, 1),
            collider_velocity: down_f() * player.jump_properties.delta_vy,
            collider_pos: p(5.0, 6.0),
            collided_block_square: p(5, 5),
        }
    }

    fn get_player_block_collision_from_running_into_wall_on_left() -> PlayerBlockCollision {
        PlayerBlockCollision {
            time_in_ticks: 0.0,
            normal: p(1, 0),
            collider_velocity: left_f() * DEFAULT_PLAYER_MAX_RUN_SPEED,
            collider_pos: p(6.0, 5.0),
            collided_block_square: p(5, 5),
        }
    }

    fn set_up_player_halfway_through_decompressing_from_vertical_jump_on_platform() -> Game {
        let mut game = set_up_player_on_platform();
        let mut collision = get_vertical_jump_landing_player_block_collision(&game.player);
        collision.time_in_ticks = game.time_in_ticks()
            - (DEFAULT_TICKS_TO_MAX_COMPRESSION + DEFAULT_TICKS_FROM_MAX_TO_END_COMPRESSION / 2.0);
        game.player.last_collision = Some(collision);
        game
    }

    fn set_up_player_fully_compressed_from_vertical_jump_on_platform() -> Game {
        let mut game = set_up_player_on_platform();
        let mut collision = get_vertical_jump_landing_player_block_collision(&game.player);
        collision.time_in_ticks = game.time_in_ticks() - DEFAULT_TICKS_TO_MAX_COMPRESSION;
        game.player.last_collision = Some(collision);
        game
    }

    fn set_up_player_fully_compressed_from_vertical_jump_while_running_right_on_platform() -> Game {
        let mut game = set_up_player_fully_compressed_from_vertical_jump_on_platform();
        game.player.vel = right_f() * game.player.max_run_speed;
        game.player
            .last_collision
            .as_mut()
            .unwrap()
            .collider_velocity
            .set_x(game.player.vel.x());
        game.player.desired_direction = right_i();
        game
    }

    fn set_up_player_fully_compressed_from_leftwards_impact_on_wall() -> Game {
        let mut game = set_up_player_hanging_on_wall_on_left();
        let mut collision = get_player_block_collision_from_running_into_wall_on_left();
        collision.time_in_ticks = game.time_in_ticks() - DEFAULT_TICKS_TO_MAX_COMPRESSION;
        game.player.last_collision = Some(collision);
        game
    }

    fn set_up_player_fully_compressed_from_down_leftwards_impact_on_wall() -> Game {
        let mut game = set_up_player_hanging_on_wall_on_left();
        let mut collision = get_player_block_collision_from_running_into_wall_on_left();
        collision
            .collider_velocity
            .set_y(-game.player.jump_properties.delta_vy);
        collision.time_in_ticks = game.time_in_ticks() - DEFAULT_TICKS_TO_MAX_COMPRESSION;
        game.player.last_collision = Some(collision);
        game.player.vel = down_f() * game.player.jump_properties.delta_vy;
        game
    }

    fn set_up_particle_passing_over_player_on_platform() -> Game {
        let mut game = set_up_player_on_platform();
        let particle_start_pos = game.player.pos + up_f() * 5.0;
        game.place_particle_with_velocity(particle_start_pos, right_f() * DEFAULT_PARTICLE_SPEED);
        game
    }

    fn set_up_one_stationary_particle() -> Game {
        let mut game = set_up_game();
        game.place_particle(p(15.0, 15.0));
        game
    }

    fn set_up_player_running_full_speed_to_right_on_platform() -> Game {
        let mut game = set_up_player_on_platform();
        game.player.vel = right_f() * game.player.max_run_speed;
        game.player.desired_direction = right_i();
        game
    }

    fn set_up_player_running_full_speed_to_left_on_platform() -> Game {
        let mut game = set_up_player_on_platform();
        game.player.vel = left_f() * game.player.max_run_speed;
        game.player.desired_direction = left_i();
        game
    }

    fn set_up_player_running_double_max_speed_left() -> Game {
        let mut game = set_up_player_on_platform();
        game.player.vel = left_f() * game.player.max_run_speed * 2.0;
        game.player.desired_direction = left_i();
        game
    }

    fn set_up_player_running_double_max_run_speed_right() -> Game {
        let mut game = set_up_player_on_platform();
        game.player.vel = right_f() * game.player.max_run_speed * 2.0;
        game.player.desired_direction = right_i();
        game
    }

    fn set_up_player_running_double_max_boost_speed_right() -> Game {
        let mut game = set_up_player_on_platform();
        game.player.vel = right_f() * game.player.dash_vel * 2.0;
        game.player.desired_direction = right_i();
        game
    }

    fn set_up_player_about_to_stop_moving_right_on_platform() -> Game {
        let mut game = set_up_player_on_platform();
        game.player.vel = right_f() * 0.001;
        game.player.desired_direction = down_i();
        game
    }

    fn set_up_player_moving_full_speed_to_right_in_space() -> Game {
        let mut game = set_up_just_player();
        be_in_space(&mut game);
        game.player.vel = right_f() * game.player.max_run_speed;
        game.player.desired_direction = p(1, 0);
        game
    }

    fn set_up_player_about_to_run_into_block_on_platform() -> Game {
        let mut game = set_up_player_running_full_speed_to_right_on_platform();
        game.place_block(round(game.player.pos) + p(2, 0), Block::Wall);
        game.player.pos.add_assign(p(0.999, 0.0));
        game
    }

    fn set_up_player_about_to_land_straight_down() -> Game {
        let mut game = set_up_just_player();
        game.player.vel = down_f() * game.player.max_run_speed; // somewhat arbitrary
        game.place_block(round(game.player.pos) + down_i() * 2, Block::Wall);
        game.player.pos.add_assign(down_f() * 0.999);
        game
    }

    fn set_up_player_about_to_hit_wall_on_right_midair() -> Game {
        let mut game = set_up_just_player();
        let start_square: IPoint = round(game.player.pos);
        let wall_x = start_square.x() + 1;
        game.place_line_of_blocks((wall_x, 0), (wall_x, game.height() as i32 - 1), Block::Wall);
        game.player.pos.add_assign(left_f() * 0.01);
        game.player.vel = right_f() * game.player.max_run_speed;
        game.player.desired_direction = right_i();
        game
    }

    fn set_up_player_about_to_hit_block_in_space() -> Game {
        let mut game = set_up_player_moving_full_speed_to_right_in_space();
        game.place_block(round(game.player.pos) + p(2, 0), Block::Wall);
        game.player.pos.add_assign(p(0.999, 0.0));
        game
    }

    fn set_up_player_t_ticks_from_hitting_a_block_in_space(t: f32) -> Game {
        let mut game = set_up_just_player();
        be_in_space(&mut game);

        game.place_block(round(game.player.pos) + right_i() * 2, Block::Wall);

        game.player.vel = right_f() / t;
        game
    }

    fn set_up_player_very_slightly_above_platform() -> Game {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(0.0, 0.001));
        return game;
    }

    fn set_up_player_on_platform_in_frictionless_vacuum() -> Game {
        let mut game = set_up_player_on_platform();
        be_in_vacuum(&mut game);
        be_frictionless(&mut game);
        return game;
    }

    fn set_up_player_boosting_through_space_in_direction(dir: Point<f32>) -> Game {
        let mut game = set_up_just_player();
        be_in_space(&mut game);
        let vel = direction(dir) * game.player.speed_of_blue * 1.1;
        game.player.desired_direction = snap_to_grid(dir);
        game.player_dash();
        game.player.vel = vel;
        game
    }

    fn set_up_player_on_block() -> Game {
        let mut game = set_up_just_player();
        let block_pos = snap_to_grid(game.player.pos) + p(0, -1);
        game.place_wall_block(block_pos);
        return game;
    }

    fn set_up_player_on_block_more_overhanging_than_not_on_right() -> Game {
        let mut game = set_up_player_on_block();
        game.player.pos.add_assign(p(0.7, 0.0));
        return game;
    }

    fn set_up_player_under_platform() -> Game {
        let mut game = set_up_just_player();
        let platform_y = game.player.pos.y() as i32 + 1;
        game.place_line_of_blocks((10, platform_y), (20, platform_y), Block::Wall);
        return game;
    }

    fn set_up_player_one_tick_from_platform_impact() -> Game {
        let mut game = set_up_player_very_slightly_above_platform();
        game.player.vel = p(0.0, -10.0);
        return game;
    }

    fn set_up_player_touching_wall_on_right() -> Game {
        let mut game = set_up_just_player();
        let wall_x = game.player.pos.x() as i32 + 1;
        game.place_line_of_blocks((wall_x, 0), (wall_x, 20), Block::Wall);
        return game;
    }

    fn set_up_player_almost_touching_wall_on_right() -> Game {
        let mut game = set_up_player_touching_wall_on_right();
        game.player.pos.add_assign(p(-0.1, 0.0));
        return game;
    }

    fn set_up_player_in_corner_of_big_L() -> Game {
        let mut game = set_up_player_on_platform();
        game.place_line_of_blocks((14, 10), (14, 20), Block::Wall);
        return game;
    }

    fn set_up_player_stationary_slightly_to_the_side_of_corner_of_big_L() -> Game {
        let mut game = set_up_player_on_platform();
        game.place_line_of_blocks((14, 10), (14, 20), Block::Wall);
        game.player.pos.add_assign(right_f() * 0.01);
        return game;
    }

    fn set_up_player_in_corner_of_backward_L() -> Game {
        let mut game = set_up_player_on_platform();
        let wall_x = game.player.pos.x() as i32 + 1;
        game.place_line_of_blocks(
            (wall_x, game.player.pos.y() as i32 - 1),
            (wall_x, game.player.pos.y() as i32 + 5),
            Block::Wall,
        );
        return game;
    }

    fn set_up_player_about_to_run_into_corner_of_backward_L() -> Game {
        let mut game = set_up_player_in_corner_of_backward_L();
        game.player.pos.add_assign(left_f() * 0.1);
        game.player.desired_direction = right_i();
        game.player.vel = right_f() * 5.0;
        return game;
    }

    fn set_up_player_starting_to_move_right_on_platform() -> Game {
        let mut game = set_up_player_on_platform();
        game.player.desired_direction = p(1, 0);
        return game;
    }

    fn set_up_player_on_platform_in_box() -> Game {
        let mut game = set_up_player_on_platform();
        game.place_boundary_wall();
        return game;
    }

    fn set_up_player_hanging_on_wall_on_left() -> Game {
        let mut game = set_up_just_player();
        let wall_x = game.player.pos.x() as i32 - 1;
        game.place_line_of_blocks((wall_x, 0), (wall_x, 20), Block::Wall);
        game.player.desired_direction.set_x(-1);
        return game;
    }

    fn set_up_player_about_to_reach_peak_of_jump() -> Game {
        let mut game = set_up_just_player();
        game.player.vel = up_f() * 0.0001;
        game
    }

    fn set_up_particle_moving_in_direction_about_to_hit_block_at_square(
        dir: Point<f32>,
        block: Block,
        square: Point<i32>,
    ) -> Game {
        let mut game = set_up_game();
        let particle_start_pos = floatify(square) + -dir * 0.51;
        let particle_start_vel = dir * 0.1;
        game.place_block(square, block);
        game.place_particle_with_velocity(particle_start_pos, particle_start_vel);
        game
    }

    fn set_up_particle_moving_in_direction_about_to_hit_wall_at_square(
        dir: Point<f32>,
        square: Point<i32>,
    ) -> Game {
        set_up_particle_moving_in_direction_about_to_hit_block_at_square(dir, Block::Wall, square)
    }

    fn set_up_particle_moving_right_and_about_to_hit_block(block: Block) -> Game {
        set_up_particle_moving_in_direction_about_to_hit_block_at_square(right_f(), block, p(5, 5))
    }

    fn set_up_particle_moving_right_and_about_to_hit_wall() -> Game {
        set_up_particle_moving_right_and_about_to_hit_block(Block::Wall)
    }

    fn set_up_particle_moving_right_and_about_to_hit_particle_amalgam() -> Game {
        set_up_particle_moving_right_and_about_to_hit_block(Block::ParticleAmalgam(5))
    }

    fn set_up_particle_moving_right_and_about_to_hit_particle_amalgam_at_explosion_threshold(
    ) -> Game {
        set_up_particle_moving_right_and_about_to_hit_block(Block::ParticleAmalgam(
            DEFAULT_PARTICLES_IN_AMALGAMATION_FOR_EXPLOSION - 1,
        ))
    }

    fn set_up_30_particles_about_to_move_one_square_right() -> Game {
        set_up_n_particles_about_to_move_one_square_right(30)
    }

    fn set_up_n_particles_about_to_move_one_square_right(n: i32) -> Game {
        let mut game = set_up_game();
        let start_pos = p(0.49, 0.0);
        let start_vel = p(0.1, 0.0);
        for _ in 0..n {
            game.place_particle_with_velocity(start_pos, start_vel);
        }
        game
    }

    fn set_up_30_particles_moving_slowly_right_from_origin() -> Game {
        let mut game = set_up_game();
        let start_pos = p(0.0, 0.0);
        let start_vel = p(0.1, 0.0);
        for _ in 0..30 {
            game.place_particle_with_velocity(start_pos, start_vel);
        }
        game
    }

    fn set_up_particles_about_to_combine() -> Game {
        let mut game = set_up_game();
        game.place_n_particles(
            game.particle_amalgamation_density,
            floatify(game.mid_square()),
        );
        for particle in &mut game.particles {
            particle.start_pos = particle.start_pos + left_f() * 2.0;
        }
        game
    }

    fn set_up_four_wall_blocks_at_5_and_6() -> Game {
        let mut game = set_up_game();
        game.place_wall_block(p(5, 5));
        game.place_wall_block(p(6, 5));
        game.place_wall_block(p(5, 6));
        game.place_wall_block(p(6, 6));
        game
    }

    fn set_up_plus_sign_wall_blocks_at_square(square: Point<i32>) -> Game {
        let mut game = set_up_game();
        game.place_wall_block(square);
        for i in 0..4 {
            game.place_wall_block(square + orthogonal_direction(i));
        }
        game
    }

    fn set_up_particle_about_to_hit_concave_corner_exactly() -> Game {
        let plus_center = p(7, 7);
        let mut game = set_up_plus_sign_wall_blocks_at_square(plus_center);
        let start_pos = floatify(plus_center) + p(-0.51, -0.51);
        let start_vel = p(0.1, 0.1);
        game.place_particle_with_velocity(start_pos, start_vel);
        game
    }

    fn set_up_particle_about_to_hit_concave_corner() -> Game {
        let plus_center = p(7, 7);
        let mut game = set_up_plus_sign_wall_blocks_at_square(plus_center);
        let start_pos = floatify(plus_center) + p(-0.51, 0.52);
        let start_vel = p(0.1, -0.1);
        game.place_particle_with_velocity(start_pos, start_vel);
        game
    }

    fn set_up_n_particles_about_to_bounce_off_platform_at_grid_bottom(n: i32) -> Game {
        let mut game = set_up_game();
        let platform_y = 0;
        let platform_start_x = 10;
        let platform_width = 10;

        game.place_line_of_blocks(
            (platform_start_x, platform_y),
            (platform_start_x + platform_width, platform_y),
            Block::Wall,
        );
        for i in 0..n {
            let mut particle = game.make_particle_with_velocity(
                p(
                    lerp(
                        platform_start_x as f32 + platform_width as f32 / 3.0,
                        platform_start_x as f32 + platform_width as f32 * 2.0 / 3.0,
                        i as f32 / n as f32,
                    ),
                    platform_y as f32 + 0.6,
                ),
                p(i as f32 / n as f32 - 0.5, -2.0),
            );
            particle.wall_collision_behavior = ParticleWallCollisionBehavior::Bounce;
            game.add_particle(particle);
        }
        return game;
    }

    fn set_up_turret_facing_up() -> Game {
        set_up_turret_facing_direction(up_f())
    }

    fn set_up_turret_facing_direction(dir: FPoint) -> Game {
        let mut game = set_up_game();
        game.place_turret(game.mid_square());
        game.turrets[0].laser_direction = dir;
        game
    }

    fn set_up_turret_aiming_at_stationary_particle() -> Game {
        let mut game = set_up_turret_facing_up();
        let turret_pos = floatify(game.turrets[0].square);
        game.place_particle(turret_pos + up_f() * 3.0);
        game
    }

    fn set_up_turret_aiming_at_particle_amalgam() -> Game {
        let mut game = set_up_turret_facing_up();
        let turret_square = game.turrets[0].square;
        game.place_block(turret_square + up_i() * 3, Block::ParticleAmalgam(50));
        game
    }

    fn set_up_single_step_foe() -> Game {
        let mut game = set_up_game();
        game.place_step_foe(game.mid_square());
        game
    }

    fn be_frictionless(game: &mut Game) {
        game.player.acceleration_from_floor_traction = 0.0;
        game.player.deceleration_from_ground_friction = 0.0;
    }

    fn be_in_vacuum(game: &mut Game) {
        game.player.acceleration_from_air_traction = 0.0;
        game.player.deceleration_from_air_friction = 0.0;
    }

    fn be_in_zero_g(game: &mut Game) {
        game.player.jump_properties.g = 0.0
    }

    fn be_in_space(game: &mut Game) {
        be_in_zero_g(game);
        be_in_vacuum(game);
    }

    fn be_in_frictionless_space(game: &mut Game) {
        be_frictionless(game);
        be_in_space(game);
    }

    #[ignore] // block gravity disabled for now
    #[test]
    #[timeout(100)]
    fn test_block_gravity() {
        let mut game = Game::new(30, 30);
        game.place_block(p(10, 10), Block::Wall);
        game.place_block(p(20, 10), Block::Brick);

        assert!(game.grid[10][10] == Block::Wall);
        assert!(game.grid[20][10] == Block::Brick);
        game.tick_physics();

        assert!(game.grid[10][10] == Block::Wall);
        assert!(game.grid[20][10] == Block::Air);

        assert!(game.grid[10][9] == Block::Air);
        assert!(game.grid[20][9] == Block::Brick);
    }

    #[test]
    #[timeout(100)]
    fn test_place_player() {
        let mut game = Game::new(30, 30);
        // TODO: should these be variables?  Or should I just hardcode them?
        let (x1, y1, x2, y2) = (15.0, 11.0, 12.0, 5.0);
        assert!(game.player.alive == false);
        game.place_player(x1, y1);

        assert!(game.player.pos == p(x1, y1));
        assert!(game.player.alive == true);

        game.place_player(x2, y2);
        assert!(game.player.pos == p(x2, y2));
        assert!(game.player.alive == true);
    }

    #[test]
    #[timeout(100)]
    fn test_player_dies_when_falling_off_screen() {
        let mut game = Game::new(30, 30);
        game.place_player(15.0, 0.0);
        game.player.jump_properties.g = 1.0;
        game.tick_physics();
        game.tick_physics();
        assert!(game.player.alive == false);
    }

    #[test]
    #[timeout(100)]
    fn test_player_falls_in_gravity() {
        let mut game = set_up_just_player();
        game.player.jump_properties.g = 1.0;
        let start_pos = game.player.pos;
        let start_vel = game.player.vel;
        game.tick_physics();
        assert!(game.player.pos.y() < start_pos.y());
        assert!(game.player.vel.y() < start_vel.y());
    }

    #[test]
    #[timeout(100)]
    fn test_single_block_squarecast_no_move() {
        let point = p(0.0, 0.0);
        let p_wall = p(5, 5);

        assert!(!single_block_unit_squarecast(point, point, p_wall).hit_something());
    }

    #[test]
    #[timeout(100)]
    fn test_squarecast_no_move() {
        let game = Game::new(30, 30);
        let point = p(0.0, 0.0);

        assert!(!game.unit_squarecast(point, point).hit_something());
    }

    #[test]
    #[timeout(100)]
    fn test_squarecast_horizontal_hit() {
        let mut game = Game::new(30, 30);
        let p_wall = p(5, 0);
        game.place_block(p_wall, Block::Wall);

        let p1 = floatify(p_wall) + p(-2.0, 0.0);
        let p2 = floatify(p_wall) + p(2.0, 0.0);
        let result = game.unit_squarecast(p1, p2);

        assert!(result.hit_something());
        assert!(points_nearly_equal(
            result.collider_pos,
            floatify(p_wall) + p(-1.0, 0.0)
        ));
        assert!(result.collision_normal.unwrap() == p(-1, 0));
    }

    #[test]
    #[timeout(100)]
    fn test_squarecast_vertical_hit() {
        let mut game = Game::new(30, 30);
        let p_wall = p(15, 10);
        game.place_block(p_wall, Block::Wall);

        let p1 = floatify(p_wall) + p(0.0, -1.1);
        let p2 = floatify(p_wall);
        let result = game.unit_squarecast(p1, p2);

        assert!(result.hit_something());
        assert!(points_nearly_equal(
            result.collider_pos,
            floatify(p_wall) + p(0.0, -1.0)
        ));
        assert!(result.collision_normal.unwrap() == p(0, -1));
    }

    #[test]
    #[timeout(100)]
    fn test_squarecast_end_slightly_overlapping_a_block() {
        let mut game = Game::new(30, 30);
        let p_wall = p(15, 10);
        game.place_block(p_wall, Block::Wall);

        let p1 = floatify(p_wall) + p(0.0, 1.5);
        let p2 = floatify(p_wall) + p(0.0, 0.999);
        let result = game.unit_squarecast(p1, p2);

        assert!(result.hit_something());
        assert!(points_nearly_equal(
            result.collider_pos,
            floatify(p_wall) + p(0.0, 1.0)
        ));
        assert!(result.collision_normal.unwrap() == p(0, 1));
    }

    #[test]
    #[timeout(100)]
    fn test_unit_squarecast_to_upper_right() {
        let mut game = Game::new(30, 30);
        game.place_line_of_blocks((10, 10), (20, 10), Block::Wall);

        let squarecast_result = game.unit_squarecast(p(15.0, 9.0), p(17.0, 11.0));
        assert!(points_nearly_equal(
            squarecast_result.collider_pos,
            p(15.0, 9.0)
        ));
        assert!(squarecast_result.collision_normal.unwrap() == p(0, -1));
        assert!(squarecast_result.collided_block_square.unwrap().y() == 10);
    }

    #[test]
    #[timeout(100)]
    fn test_unit_squarecast_to_right() {
        let mut game = Game::new(30, 30);
        let wall_square = p(10, 10);
        game.place_wall_block(wall_square);

        let squarecast_result = game.unit_squarecast(
            floatify(wall_square + p(-5, 0)),
            floatify(wall_square + p(5, 0)),
        );
        assert!(points_nearly_equal(
            squarecast_result.collider_pos,
            floatify(wall_square) - p(1.0, 0.0)
        ));
        assert!(squarecast_result.collision_normal.unwrap() == p(-1, 0));
        assert!(squarecast_result.collided_block_square.unwrap().y() == wall_square.y());
    }

    #[test]
    #[timeout(100)]
    fn test_unit_squarecast__move_up_away_from_exactly_touching() {
        let mut game = Game::new(30, 30);
        let wall_square = p(10, 10);
        game.place_wall_block(wall_square);

        let squarecast_result = game.unit_squarecast(
            floatify(wall_square + p(0, 1)),
            floatify(wall_square + p(0, 5)),
        );
        assert!(!squarecast_result.hit_something());
    }

    #[test]
    #[timeout(100)]
    fn test_unit_squarecast_to_far_upper_right() {
        let mut game = Game::new(30, 30);
        game.place_line_of_blocks((10, 10), (20, 10), Block::Wall);
        let squarecast_result = game.unit_squarecast(p(15.0, 9.0), p(17.0, 110.0));
        assert!(points_nearly_equal(
            squarecast_result.collider_pos,
            p(15.0, 9.0)
        ));
        assert!(squarecast_result.collision_normal.unwrap() == p(0, -1));
        assert!(squarecast_result.collided_block_square.unwrap().y() == 10);

        assert!(!game
            .unit_squarecast(p(1.0, 9.0), p(-17.0, 9.0))
            .hit_something());
        assert!(!game
            .unit_squarecast(p(15.0, 9.0), p(17.0, -11.0))
            .hit_something());
    }

    #[test]
    #[timeout(100)]
    fn test_squarecast_skips_player() {
        let game = set_up_player_on_platform();

        assert!(!game
            .unit_squarecast(p(15.0, 11.0), p(15.0, 13.0))
            .hit_something());
    }

    #[test]
    #[timeout(100)]
    fn test_in_world_check() {
        let game = Game::new(30, 30);
        assert!(game.square_is_in_world(p(0, 0)));
        assert!(game.square_is_in_world(p(29, 29)));
        assert!(!game.square_is_in_world(p(30, 30)));
        assert!(!game.square_is_in_world(p(10, -1)));
        assert!(!game.square_is_in_world(p(-1, 10)));
    }

    #[test]
    #[timeout(100)]
    fn test_move_player_right() {
        let mut game = set_up_player_on_platform();
        game.player.max_run_speed = 1.0;
        game.player.acceleration_from_floor_traction = 1.0;
        game.player.desired_direction = right_i();
        let start_pos = game.player.pos;
        let start_vel = game.player.vel;

        game.tick_physics();

        assert!(game.player.pos == start_pos + right_f() * 0.5);
        assert!(game.player.vel == start_vel + right_f() * 1.0);
    }

    #[test]
    #[timeout(100)]
    fn test_move_player_left() {
        let mut game = set_up_player_on_platform();
        game.place_player(15.0, 11.0);
        game.player.max_run_speed = 1.0;
        game.player.acceleration_from_floor_traction = 1.0;
        let start_pos = game.player.pos;
        let start_vel = game.player.vel;

        assert!(game.player.desired_direction.x() == 0);
        game.player.desired_direction = left_i();

        game.tick_physics();

        assert!(game.player.pos == start_pos + left_f() * 0.5);
    }

    #[test]
    #[timeout(100)]
    fn test_move_player_for_multiple_ticks() {
        let mut game = set_up_player_on_platform();
        game.player.max_run_speed = 1.0;
        game.player.ground_friction_start_speed = game.player.max_run_speed;
        game.player.acceleration_from_floor_traction = 1.0;
        let start_pos = game.player.pos;
        let start_vel = game.player.vel;

        assert!(game.player.desired_direction.x() == 0);
        game.player.desired_direction = left_i();

        let num_ticks = 5;

        for _ in 0..num_ticks {
            game.tick_physics();
        }

        assert!(game.player.pos == start_pos + left_f() * (num_ticks as f32 - 0.5));
        assert!(game.player.vel == start_vel + left_f() * game.player.max_run_speed);
    }

    #[test]
    #[timeout(100)]
    fn test_player_can_stop_from_sprint() {
        let mut game = set_up_player_running_full_speed_to_right_on_platform();
        game.player.desired_direction = down_i();
        let start_vel = game.player.vel;

        let ticks_to_stop = ticks_to_stop_from_speed(
            start_vel.x().abs(),
            game.player.acceleration_from_floor_traction,
        )
        .unwrap()
        .ceil() as i32;

        assert!(ticks_to_stop < 500);

        let mut prev_x_vel = game.player.vel.x();
        for _ in 0..ticks_to_stop - 1 {
            game.tick_physics();
            assert!(game.player.vel.x() < prev_x_vel);
            prev_x_vel = game.player.vel.x();
        }
        // one last tick just in case
        game.tick_physics();
        assert!(game.player.vel.x() == 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_player_can_come_to_full_stop() {
        let mut game = set_up_player_about_to_stop_moving_right_on_platform();

        game.tick_physics();

        assert!(game.player.vel == zero_f());
    }

    #[test]
    #[timeout(100)]
    fn test_player_only_comes_to_full_stop_if_desired() {
        let mut game1 = set_up_player_about_to_stop_moving_right_on_platform();
        let mut game2 = set_up_player_about_to_stop_moving_right_on_platform();

        game1.player.desired_direction = down_i();
        game2.player.desired_direction = left_i();

        assert!(game1.player_wants_to_come_to_a_full_stop());
        assert!(!game2.player_wants_to_come_to_a_full_stop());

        game1.tick_physics();
        game2.tick_physics();

        assert!(game1.player.vel.x() == 0.0);
        assert!(game2.player.vel.x() < 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_stop_on_collision() {
        let mut game = set_up_player_about_to_run_into_block_on_platform();
        let mut game = set_up_player_on_platform();
        let start_pos = game.player.pos;
        let good_end_pos = floatify(snap_to_grid(start_pos));

        game.tick_physics();

        assert!(game.player.pos == good_end_pos);
        assert!(game.player.vel == zero_f());
    }

    #[test]
    #[timeout(100)]
    fn test_stop_on_collision_in_space() {
        let mut game = set_up_player_about_to_hit_block_in_space();
        let start_pos = game.player.pos;
        let good_end_pos = floatify(snap_to_grid(start_pos));

        game.tick_physics();

        assert!(game.player.pos == good_end_pos);
        assert!(game.player.vel == zero_f());
    }

    #[test]
    #[timeout(100)]
    fn test_snap_to_object_on_collision() {
        let mut game = set_up_player_about_to_run_into_block_on_platform();
        let starting_square = snap_to_grid(game.player.pos);
        let block_square = starting_square + p(1, 0);

        game.tick_physics();

        assert!(game.player.pos == floatify(starting_square));
        assert!(game.get_block(block_square) == Block::Wall);
    }

    #[test]
    #[timeout(100)]
    fn test_player_does_not_accelerate_past_run_speed_cap() {
        let mut game = set_up_player_on_platform();
        game.player.max_run_speed = 0.49;
        game.player.acceleration_from_floor_traction = 999.9; // rilly fast
        game.player.desired_direction.set_x(1);

        game.tick_physics();

        assert!(game.player.pos.x() > 15.0);
        assert!(game.player.pos.x() < 15.5);

        game.tick_physics();

        assert!(game.player.pos.x() > 15.5);
        assert!(game.player.pos.x() < 16.0);
    }

    #[test]
    #[timeout(100)]
    fn test_move_player_quickly() {
        let mut game = set_up_player_starting_to_move_right_on_platform();
        game.player.max_run_speed = 2.0;
        game.player.acceleration_from_floor_traction = 1.0;
        game.player.ground_friction_start_speed = 5.0;
        let start_pos = game.player.pos;
        let start_vel = game.player.vel;

        game.tick_physics();
        game.tick_physics();
        assert!(game.player.pos.x() == (start_pos + right_f() * 2.0).x());
        assert!(game.player.vel.x() == (start_vel + right_f() * 2.0).x());
        game.tick_physics();
        assert!(game.player.vel.x() == (start_vel + right_f() * 2.0).x());
        assert!(game.player.pos.x() == (start_pos + right_f() * 4.0).x());
    }

    #[test]
    #[timeout(100)]
    fn test_fast_player_collision_between_ticks() {
        let mut game = set_up_player_on_platform();
        // Player should not teleport through this block
        game.place_block(p(16, 11), Block::Wall);
        game.player.max_run_speed = 2.0;
        game.player.desired_direction.set_x(1);

        game.tick_physics();
        assert!(game.player.pos == p(15.0, 11.0));
        assert!(game.player.vel.x() == 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_can_jump() {
        let mut game = set_up_player_on_platform();
        let start_pos = game.player.pos;

        game.player_jump();
        game.tick_physics();
        assert!(game.player.pos.y() > start_pos.y());
    }

    #[test]
    #[timeout(100)]
    fn test_player_gravity() {
        let mut game = Game::new(30, 30);
        game.place_player(15.0, 11.0);
        game.player.jump_properties.g = 1.0;

        game.tick_physics();

        assert!(game.player.pos.y() < 11.0);
    }

    #[test]
    #[timeout(100)]
    fn test_land_after_jump__vertical() {
        let mut game = set_up_player_on_platform();
        let start_pos = game.player.pos;

        game.player_jump();
        for _ in 0..50 {
            game.tick_physics();
            // dbg!( "a", game.player.pos.y(), game.player.vel.y(), game.player.accel.y() );
        }
        assert!(nearly_equal(game.player.pos.y(), start_pos.y()));
    }

    #[test]
    #[timeout(100)]
    fn test_land_after_jump__moving_right() {
        let mut game = set_up_player_running_full_speed_to_right_on_platform();
        game.set_jump_duration_in_seconds_and_height_in_grid_squares(0.1, 0.1);
        be_in_vacuum(&mut game);
        let start_pos = game.player.pos;

        game.player_jump();
        let mut prev_vel_y = game.player.vel.y();
        let jump_duration_in_ticks: i32 = game.player.jump_properties.duration.ceil() as i32;
        for _ in 0..jump_duration_in_ticks {
            game.tick_physics();
            assert!(game.player.vel.x().abs() == game.player.max_run_speed);
            assert!(game.player.accel.x() == 0.0);
            //dbg!(game.player.kinematic_state());
            assert!(game.player.vel.y() < prev_vel_y || game.player.vel.y() == 0.0);
            prev_vel_y = game.player.vel.y();
        }
        assert!(nearly_equal(game.player.pos.y(), start_pos.y()));
    }

    #[test]
    #[timeout(100)]
    fn test_land_after_jump__moving_left() {
        let mut game = set_up_player_running_full_speed_to_left_on_platform();
        game.set_jump_duration_in_seconds_and_height_in_grid_squares(0.1, 0.1);
        be_in_vacuum(&mut game);
        let start_pos = game.player.pos;

        game.player_jump();
        let initial_vy = game.player.vel.y();
        let mut prev_vel_y = game.player.vel.y();
        let mut initial_y = game.player.pos.y();
        let jump_duration_in_ticks: i32 = game.player.jump_properties.duration.ceil() as i32;
        for _ in 0..jump_duration_in_ticks {
            game.tick_physics();
            assert!(game.player.vel.x().abs() == game.player.max_run_speed);
            assert!(game.player.accel.x() == 0.0);
            assert!(game.player.vel.y() < prev_vel_y || game.player.vel.y() == 0.0);
            assert!(game.player.vel.y().abs() <= initial_vy.abs());
            assert!(game.player.pos.y() >= initial_y);
            prev_vel_y = game.player.vel.y();
        }
        assert!(nearly_equal(game.player.pos.y(), start_pos.y()));
    }

    #[test]
    #[timeout(100)]
    fn test_land_on_ground() {
        let mut game = set_up_player_about_to_land_straight_down();
        let start_pos = game.player.pos;
        let good_end_pos = floatify(snap_to_grid(start_pos));

        game.tick_physics();
        //dbg!(game.player.pos, game.player.vel);
        assert!(game.player.pos == good_end_pos);
        assert!(game.player.vel == zero_f());
    }

    #[test]
    #[timeout(100)]
    fn test_slide_on_angled_collision() {
        let mut game = Game::new(30, 30);
        game.place_line_of_blocks((10, 10), (20, 10), Block::Wall);
        game.place_line_of_blocks((14, 10), (14, 20), Block::Wall);

        game.place_player(15.1, 11.0);
        game.player.vel = p(-2.0, 2.0);

        game.update_output_buffer();
        game.print_output_buffer();

        game.tick_physics();
        assert!(game.player.vel.x() == 0.0);
        assert!(game.player.vel.y() > 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_decellerate_when_already_moving_faster_than_max_speed() {
        let mut game = set_up_player_on_platform();
        game.player.max_run_speed = 1.0;
        game.player.vel.set_x(5.0);
        game.player.desired_direction.set_x(1);
        game.tick_physics();
        assert!(game.player.vel.x() > game.player.max_run_speed);
        assert!(game.player.vel.x() < 5.0);
    }

    #[test]
    #[timeout(100)]
    fn test_no_double_jump() {
        let mut game = set_up_player_on_platform();
        game.player_jump_if_possible();
        game.tick_physics();
        let vel_y_before_second_jump = game.player.vel.y();
        game.player_jump_if_possible();
        game.tick_physics();
        assert!(game.player.vel.y() < vel_y_before_second_jump);
    }

    #[test]
    #[timeout(100)]
    fn test_respawn_button() {
        let mut game = set_up_game();
        game.handle_event(Event::Key(Key::Char('r')));
        assert!(game.player.alive);
        assert!(game.player.pos == p(15.0, 15.0));
    }

    #[test]
    #[timeout(100)]
    fn test_wall_grab() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.tick_physics();
        assert!(game.player.vel.y() == 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_simple_wall_jump() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_jump_if_possible();
        game.tick_physics();
        assert!(game.player.vel.y() > 0.0);
        assert!(game.player.vel.x() > 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_no_running_up_walls_immediately_after_spawn() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player.vel = p(-1.0, 1.0);
        let start_vel_y = game.player.vel.y();
        game.tick_physics();
        assert!(game.player.vel.y() < start_vel_y);
    }

    #[test]
    #[timeout(100)]
    fn test_dont_grab_wall_while_moving_up() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player.vel.set_y(1.0);
        assert!(!game.player_is_grabbing_wall());
    }

    #[test]
    #[timeout(100)]
    fn test_no_friction_when_sliding_up_wall() {
        let mut game = set_up_player_on_platform();
        let wall_x = game.player.pos.x() as i32 - 2;
        game.place_line_of_blocks((wall_x, 0), (wall_x, 20), Block::Wall);
        game.player.desired_direction.set_x(-1);
        game.player.pos.add_assign(p(0.0, 2.0));
        game.player.pos.set_x(wall_x as f32 + 1.0);
        game.player.vel = p(-3.0, 3.0);
        game.player.jump_properties.g = 0.0;
        game.player.deceleration_from_air_friction = 0.0;

        let start_y_vel = game.player.vel.y();
        let start_y_pos = game.player.pos.y();
        game.tick_physics();
        let end_y_vel = game.player.vel.y();
        let end_y_pos = game.player.pos.y();

        assert!(start_y_vel == end_y_vel);
        assert!(start_y_pos != end_y_pos);
    }

    #[test]
    #[ignore] // TODO: test compared to naive gravity
    #[timeout(100)]
    fn test_coyote_ticks() {
        let mut game = set_up_just_player();
        let start_pos = game.player.pos;
        game.player.jump_properties.g = 1.0;
        game.tick_physics();
        assert!(game.player.pos.y() == start_pos.y());
        game.tick_physics();
        assert!(game.player.pos.y() < start_pos.y());
    }

    #[test]
    #[ignore] // TODO: test compared to naive gravity
    #[timeout(100)]
    fn test_coyote_ticks_dont_assist_jump() {
        let mut game1 = set_up_player_on_platform();
        let mut game2 = set_up_player_on_platform();

        game1.player_jump_if_possible();
        game2.player_jump_if_possible();
        game1.tick_physics();
        game2.tick_physics();

        // Second tick after jump is where coyote physics may come into play
        game1.tick_physics();
        game2.tick_physics();

        assert!(game1.player.vel.y() == game2.player.vel.y());
    }

    #[test]
    #[timeout(100)]
    fn test_wall_jump_while_starting_to_run_up_wall() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player.vel.set_y(1.0);
        game.player_jump_if_possible();
        assert!(game.player.vel.x() > 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_wall_jump_while_running_up_wall() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player.vel.set_y(1.0);
        game.tick_physics();
        assert!(game.player.vel.x() == 0.0);
        assert!(game.player.vel.y() > 0.0);
        game.player_jump_if_possible();
        assert!(game.player.vel.x() > 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_wall_jump_while_running_up_wall_after_running_into_it() {
        let mut game = set_up_player_about_to_run_into_corner_of_backward_L();
        game.internal_corner_behavior = InternalCornerBehavior::StopPlayer;
        game.player.enable_jump_compression_bonus = false;
        let away_from_wall = left_f();
        game.tick_physics();
        assert!(game.player.vel == zero_f());
        assert!(game.player.pos == floatify(snap_to_grid(game.player.pos)));
        game.player_jump_if_possible();
        dbg!(game.player.kinematic_state());
        assert!(game.player.vel.y() > 0.0);
        assert!(game.player.vel.x() == 0.0);
        game.tick_physics();
        game.tick_physics();
        game.tick_physics();
        assert!(game.player.vel.y() > 0.0);
        assert!(game.player.vel.x() == 0.0);
        game.player_jump_if_possible();
        assert!(game.player.vel.y() > 0.0);
        assert!(game.player.vel.dot(away_from_wall) > 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_wall_jump_while_jumping_beside_wall() {
        let mut game = set_up_player_in_corner_of_big_L();
        game.internal_corner_behavior = InternalCornerBehavior::StopPlayer;
        let away_from_wall = right_f();
        assert!(game.player.vel == zero_f());
        assert!(game.player.pos == floatify(snap_to_grid(game.player.pos)));
        game.player_jump_if_possible();
        assert!(game.player.vel.x() == 0.0);
        assert!(game.player.vel.y() > 0.0);
        game.tick_physics();
        game.tick_physics();
        game.tick_physics();
        assert!(game.player.vel.y() > 0.0);
        assert!(game.player.vel.x() == 0.0);
        game.player_jump_if_possible();
        assert!(game.player.vel.y() > 0.0);
        assert!(game.player.vel.dot(away_from_wall) > 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_wall_jump_behavior_for_desired_direction() {
        for wall_jump_behavior in WallJumpBehavior::iter() {
            let mut game = set_up_player_hanging_on_wall_on_left();
            let initial_wall_direction = left_i();
            game.player.wall_jump_behavior = wall_jump_behavior;
            assert!(game.player.vel == zero_f());
            game.player_jump_if_possible();
            assert!(game.player.vel != zero_f());
            let correct_final_desired_direction = match wall_jump_behavior {
                WallJumpBehavior::SwitchDirection => -initial_wall_direction,
                WallJumpBehavior::Stop => zero_i(),
                WallJumpBehavior::KeepDirection => initial_wall_direction,
            };
            assert!(game.player.desired_direction == correct_final_desired_direction);
        }
    }

    #[test]
    #[timeout(100)]
    fn test_draw_to_output_buffer() {
        let mut game = set_up_player_on_platform();
        game.update_output_buffer();
        assert!(
            game.get_buffered_glyph(snap_to_grid(game.player.pos))
                .character
                == EIGHTH_BLOCKS_FROM_LEFT[8]
        );
        assert!(
            game.get_buffered_glyph(snap_to_grid(game.player.pos + p(0.0, -1.0)))
                .character
                == Block::Wall.character()
        );
    }

    #[test]
    #[timeout(100)]
    fn test_horizontal_sub_glyph_positioning_on_left() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(-0.2, 0.0));
        game.update_output_buffer();

        let left_glyph = game.get_buffered_glyph(snap_to_grid(game.player.pos + p(-1.0, 0.0)));
        let right_glyph = game.get_buffered_glyph(snap_to_grid(game.player.pos));
        assert!(left_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[7]);
        assert!(right_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[7]);
    }

    #[test]
    #[timeout(100)]
    fn test_horizontal_sub_glyph_positioning_on_left_above_rounding_point() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(-0.49, 0.0));
        game.update_output_buffer();

        let left_glyph = game.get_buffered_glyph(snap_to_grid(game.player.pos + p(-1.0, 0.0)));
        let right_glyph = game.get_buffered_glyph(snap_to_grid(game.player.pos));
        assert!(left_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[5]);
        assert!(right_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[5]);
    }

    #[test]
    #[timeout(100)]
    fn test_horizontal_sub_glyph_positioning_on_left_rounding_down() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(-0.5, 0.0));
        game.update_output_buffer();

        let left_glyph = game.get_buffered_glyph(snap_to_grid(game.player.pos + p(-1.0, 0.0)));
        let right_glyph = game.get_buffered_glyph(snap_to_grid(game.player.pos));
        assert!(left_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[4]);
        assert!(right_glyph.character == EIGHTH_BLOCKS_FROM_LEFT[4]);
    }

    #[test]
    #[timeout(100)]
    fn test_vertical_sub_glyph_positioning_upwards() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(0.0, 0.49));
        game.update_output_buffer();

        let top_glyph = game.get_buffered_glyph(snap_to_grid(game.player.pos + p(0.0, 1.0)));
        let bottom_glyph = game.get_buffered_glyph(snap_to_grid(game.player.pos));
        assert!(top_glyph.character == EIGHTH_BLOCKS_FROM_BOTTOM[4]);
        assert!(top_glyph.fg_color == PLAYER_COLOR);
        assert!(top_glyph.bg_color == ColorName::Black);
        assert!(bottom_glyph.character == EIGHTH_BLOCKS_FROM_BOTTOM[4]);
        assert!(bottom_glyph.fg_color == ColorName::Black);
        assert!(bottom_glyph.bg_color == PLAYER_COLOR);
    }

    #[test]
    #[timeout(100)]
    fn test_vertical_sub_glyph_positioning_downwards() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(0.0, -0.2));
        game.update_output_buffer();

        let top_glyph = game.get_buffered_glyph(snap_to_grid(game.player.pos));
        let bottom_glyph = game.get_buffered_glyph(snap_to_grid(game.player.pos + p(0.0, -1.0)));
        assert!(top_glyph.character == EIGHTH_BLOCKS_FROM_BOTTOM[7]);
        assert!(top_glyph.fg_color == PLAYER_COLOR);
        assert!(top_glyph.bg_color == ColorName::Black);
        assert!(bottom_glyph.character == EIGHTH_BLOCKS_FROM_BOTTOM[7]);
        assert!(bottom_glyph.fg_color == ColorName::Black);
        assert!(bottom_glyph.bg_color == PLAYER_COLOR);
    }

    #[test]
    #[timeout(100)]
    fn test_player_glyph_when_rounding_to_zero_for_both_axes() {
        let mut game = set_up_just_player();
        game.player.pos.add_assign(p(-0.24, 0.01));
        assert!(
            game.get_player_glyphs()
                == Glyph::get_smooth_horizontal_glyphs_for_colored_floating_square(
                    game.player.pos,
                    PLAYER_COLOR
                )
        );
    }

    #[test]
    #[timeout(100)]
    fn test_player_glyphs_when_rounding_to_zero_for_x_and_half_step_up_for_y() {
        let mut game = set_up_just_player();
        game.player.pos.add_assign(p(0.24, 0.26));
        assert!(
            game.get_player_glyphs()
                == Glyph::get_smooth_vertical_glyphs_for_colored_floating_square(
                    game.player.pos,
                    PLAYER_COLOR
                )
        );
    }

    #[test]
    #[timeout(100)]
    fn test_player_glyphs_when_rounding_to_zero_for_x_and_exactly_half_step_up_for_y() {
        let mut game = set_up_just_player();
        game.player.pos.add_assign(p(0.24, 0.25));
        assert!(
            game.get_player_glyphs()
                == Glyph::get_smooth_vertical_glyphs_for_colored_floating_square(
                    game.player.pos,
                    PLAYER_COLOR
                )
        );
    }

    #[test]
    #[timeout(100)]
    fn test_player_glyphs_when_rounding_to_zero_for_x_and_exactly_half_step_down_for_y() {
        let mut game = set_up_just_player();
        game.player.pos.add_assign(p(-0.2, -0.25));
        assert!(
            game.get_player_glyphs()
                == Glyph::get_smooth_vertical_glyphs_for_colored_floating_square(
                    game.player.pos,
                    PLAYER_COLOR
                )
        );
    }

    #[test]
    #[timeout(100)]
    fn test_player_glyphs_when_rounding_to_zero_for_y_and_half_step_right_for_x() {
        let mut game = set_up_just_player();
        game.player.pos.add_assign(p(0.3, 0.1));
        assert!(
            game.get_player_glyphs()
                == Glyph::get_smooth_horizontal_glyphs_for_colored_floating_square(
                    game.player.pos,
                    PLAYER_COLOR
                )
        );
    }

    #[test]
    #[timeout(100)]
    fn test_player_glyphs_when_rounding_to_zero_for_y_and_half_step_left_for_x() {
        let mut game = set_up_just_player();
        game.player.pos.add_assign(p(-0.3, 0.2));
        assert!(
            game.get_player_glyphs()
                == Glyph::get_smooth_horizontal_glyphs_for_colored_floating_square(
                    game.player.pos,
                    PLAYER_COLOR
                )
        );
    }

    #[test]
    #[timeout(100)]
    fn test_player_glyphs_for_half_step_up_and_right() {
        let mut game = set_up_just_player();
        game.player.pos.add_assign(p(0.3, 0.4));
        let glyphs = game.get_player_glyphs();

        assert!(glyphs[0][0] == None);
        assert!(glyphs[0][1] == None);
        assert!(glyphs[0][2] == None);
        assert!(glyphs[1][0] == None);
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((1, 1)));
        assert!(glyphs[1][2].clone().unwrap().character == quarter_block_by_offset((1, -1)));
        assert!(glyphs[2][0] == None);
        assert!(glyphs[2][1].clone().unwrap().character == quarter_block_by_offset((-1, 1)));
        assert!(glyphs[2][2].clone().unwrap().character == quarter_block_by_offset((-1, -1)));
    }

    #[test]
    #[timeout(100)]
    fn test_player_glyphs_for_half_step_up_and_left() {
        let mut game = set_up_just_player();
        game.player.pos.add_assign(p(-0.4, 0.26));
        let glyphs = game.get_player_glyphs();
        assert!(glyphs[0][0] == None);
        assert!(glyphs[0][1].clone().unwrap().character == quarter_block_by_offset((1, 1)));
        assert!(glyphs[0][2].clone().unwrap().character == quarter_block_by_offset((1, -1)));
        assert!(glyphs[1][0] == None);
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((-1, 1)));
        assert!(glyphs[1][2].clone().unwrap().character == quarter_block_by_offset((-1, -1)));
        assert!(glyphs[2][0] == None);
        assert!(glyphs[2][1] == None);
        assert!(glyphs[2][2] == None);
    }

    #[test]
    #[timeout(100)]
    fn test_player_glyphs_for_half_step_down_and_left() {
        let mut game = set_up_just_player();
        game.player.pos.add_assign(p(-0.26, -0.4999));
        let glyphs = game.get_player_glyphs();
        assert!(glyphs[0][0].clone().unwrap().character == quarter_block_by_offset((1, 1)));
        assert!(glyphs[0][1].clone().unwrap().character == quarter_block_by_offset((1, -1)));
        assert!(glyphs[0][2] == None);
        assert!(glyphs[1][0].clone().unwrap().character == quarter_block_by_offset((-1, 1)));
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((-1, -1)));
        assert!(glyphs[1][2] == None);
        assert!(glyphs[2][0] == None);
        assert!(glyphs[2][1] == None);
        assert!(glyphs[2][2] == None);
    }

    #[test]
    #[timeout(100)]
    fn test_player_glyphs_for_half_step_down_and_right() {
        let mut game = set_up_just_player();
        game.player.pos.add_assign(p(0.26, -0.4));
        let glyphs = game.get_player_glyphs();
        assert!(glyphs[0][0] == None);
        assert!(glyphs[0][1] == None);
        assert!(glyphs[0][2] == None);
        assert!(glyphs[1][0].clone().unwrap().character == quarter_block_by_offset((1, 1)));
        assert!(glyphs[1][1].clone().unwrap().character == quarter_block_by_offset((1, -1)));
        assert!(glyphs[1][2] == None);
        assert!(glyphs[2][0].clone().unwrap().character == quarter_block_by_offset((-1, 1)));
        assert!(glyphs[2][1].clone().unwrap().character == quarter_block_by_offset((-1, -1)));
        assert!(glyphs[2][2] == None);
    }

    #[test]
    #[timeout(100)]
    fn test_off_alignment_player_coarse_rendering_given_diagonal_offset() {
        let mut game = set_up_just_player();

        game.player.pos.add_assign(p(0.4, -0.3));
        game.update_output_buffer();
        let top_left_glyph = game.get_buffered_glyph(snap_to_grid(game.player.pos));
        let top_right_glyph = game.get_buffered_glyph(snap_to_grid(game.player.pos + p(1.0, 0.0)));
        let bottom_left_glyph =
            game.get_buffered_glyph(snap_to_grid(game.player.pos + p(0.0, -1.0)));
        let bottom_right_glyph =
            game.get_buffered_glyph(snap_to_grid(game.player.pos + p(1.0, -1.0)));

        assert!(top_left_glyph.character == quarter_block_by_offset((1, -1)));
        assert!(top_right_glyph.character == quarter_block_by_offset((-1, -1)));
        assert!(bottom_left_glyph.character == quarter_block_by_offset((1, 1)));
        assert!(bottom_right_glyph.character == quarter_block_by_offset((-1, 1)));
    }

    #[test]
    #[timeout(100)]
    // The timing of when to slide to a stop should give the player precision positioning
    fn test_dont_snap_to_grid_when_sliding_to_a_halt() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(0.1, 0.0));
        game.player.vel = p(0.1, 0.0);
        game.player.acceleration_from_floor_traction = 0.1;
        game.player.desired_direction = p(0, 0);

        game.tick_physics();
        assert!(game.player.vel.x() == 0.0);
        assert!(offset_from_grid(game.player.pos).x() != 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_dash_no_direction_does_nothing() {
        let mut game = set_up_player_on_platform();
        let start_vel = game.player.vel;
        game.player_dash();
        assert!(game.player.vel == start_vel);
    }

    #[test]
    #[timeout(100)]
    fn test_dash_right_on_ground() {
        let mut game = set_up_player_on_platform();
        game.player.desired_direction = p(1, 0);
        game.player_dash();
        assert!(game.player.vel == floatify(game.player.desired_direction) * game.player.dash_vel);
    }

    #[ignore] // TODO: Come back to this
    #[test]
    #[timeout(100)]
    fn test_slow_down_to_exactly_max_speed_horizontally_midair() {
        let mut game = set_up_player_barely_fighting_air_friction_to_the_right_in_zero_g();
        game.tick_physics();
        assert!(game.player.vel.x() == game.player.air_friction_start_speed);
    }

    #[ignore] // TODO: Come back to this
    #[test]
    #[timeout(100)]
    fn test_slow_down_to_exactly_max_speed_vertically_midair() {
        let mut game = set_up_player_barely_fighting_air_friction_up_in_zero_g();
        dbg!(game.player.kinematic_state());
        dbg!(game.player.air_friction_start_speed);
        dbg!(game.player.deceleration_from_air_friction);
        game.tick_physics();
        dbg!(game.player.kinematic_state());
        assert!(game.player.vel.y() == game.player.air_friction_start_speed);
    }

    #[test]
    #[timeout(100)]
    fn test_do_not_grab_wall_while_standing_on_ground() {
        let mut game = set_up_player_on_platform();
        let wall_x = game.player.pos.x() as i32 - 1;
        game.place_line_of_blocks((wall_x, 0), (wall_x, 20), Block::Wall);
        game.player.desired_direction = p(-1, 0);

        assert!(game.player_is_standing_on_block() == true);
        assert!(game.player_is_grabbing_wall() == false);
    }

    #[test]
    #[timeout(100)]
    fn test_direction_buttons() {
        let mut game = set_up_player_on_platform();
        game.handle_event(Event::Key(Key::Char('a')));
        assert!(game.player.desired_direction == p(-1, 0));
        game.handle_event(Event::Key(Key::Char('s')));
        assert!(game.player.desired_direction == p(0, -1));
        game.handle_event(Event::Key(Key::Char('d')));
        assert!(game.player.desired_direction == p(1, 0));
        game.handle_event(Event::Key(Key::Char('w')));
        assert!(game.player.desired_direction == p(0, 1));

        game.handle_event(Event::Key(Key::Left));
        assert!(game.player.desired_direction == p(-1, 0));
        game.handle_event(Event::Key(Key::Down));
        assert!(game.player.desired_direction == p(0, -1));
        game.handle_event(Event::Key(Key::Right));
        assert!(game.player.desired_direction == p(1, 0));
        game.handle_event(Event::Key(Key::Up));
        assert!(game.player.desired_direction == p(0, 1));
    }

    #[ignore]
    // switching to particle bursts
    #[test]
    #[timeout(100)]
    fn test_different_color_when_go_fast() {
        let mut game = set_up_player_on_platform();
        let stopped_color = game.get_player_glyphs()[1][1].clone().unwrap().fg_color;
        game.player.vel = p(game.player.max_run_speed + 5.0, 0.0);
        let fast_color = game.get_player_glyphs()[1][1].clone().unwrap().fg_color;

        assert!(stopped_color != fast_color);
    }

    #[test]
    #[timeout(100)]
    fn test_draw_visual_braille_line_without_rounding() {
        let mut game = set_up_player_on_platform();
        let start = game.player.pos + p(0.51, 0.1);
        let end = start + p(2.1, 0.1);
        let color = ColorName::Blue;

        // Expected braille:
        // 00 00 00
        // 11 11 10
        // 00 00 00
        // 00 00 00

        game.draw_visual_braille_line(start, end, color);

        assert!(game.get_buffered_glyph(snap_to_grid(start)).fg_color == color);
        assert!(game.get_buffered_glyph(snap_to_grid(start)).character == '\u{2812}');
        assert!(
            game.get_buffered_glyph(snap_to_grid(start) + p(1, 0))
                .fg_color
                == color
        );
        assert!(
            game.get_buffered_glyph(snap_to_grid(start) + p(1, 0))
                .character
                == '\u{2812}' // '⠒'
        );
        assert!(
            game.get_buffered_glyph(snap_to_grid(start) + p(2, 0))
                .fg_color
                == color
        );
        assert!(
            game.get_buffered_glyph(snap_to_grid(start) + p(2, 0))
                .character
                == '\u{2802}'
        );
    }

    #[test]
    #[timeout(100)]
    fn test_drawn_braille_adds_instead_of_overwrites() {
        let mut game = set_up_game();
        let start_square = p(12, 5);
        let color = ColorName::Blue;
        let start_upper_line = floatify(start_square) + p(-0.1, 0.1);
        let end_upper_line = floatify(start_square) + p(1.6, 0.1);
        let start_lower_line = floatify(start_square) + p(-0.1, -0.3);
        let end_lower_line = floatify(start_square) + p(1.6, -0.3);

        game.draw_visual_braille_line(start_upper_line, end_upper_line, color);
        game.draw_visual_braille_line(start_lower_line, end_lower_line, color);

        // Expected braille:
        // 00 00 00
        // 11 11 10
        // 00 00 00
        // 11 11 10

        // \u{28D2}, \u{28D2}, \u{2842},

        assert!(game.get_buffered_glyph(start_square + p(0, 0)).character == '\u{28D2}');
        assert!(game.get_buffered_glyph(start_square + p(1, 0)).character == '\u{28D2}');
        assert!(game.get_buffered_glyph(start_square + p(2, 0)).character == '\u{2842}');
    }

    #[test]
    #[timeout(100)]
    fn test_braille_point() {
        let mut game = set_up_game();
        // Expected braille:
        // 00 01 00
        // 00 00 00
        // 00 00 00
        // 00 00 00

        // 00 00 00
        // 00 00 00
        // 00 00 00
        // 00 00 00

        game.draw_visual_braille_point(p(1.0, 1.4), ColorName::Blue);
        assert!(game.get_buffered_glyph(p(1, 1)).character == '\u{2808}');
    }

    #[ignore]
    #[test]
    #[timeout(100)]
    fn test_braille_left_behind_when_go_fast() {
        let mut game = set_up_player_on_platform_in_box();
        game.player.vel = p(game.player.speed_of_blue * 5.0, 0.0);
        game.player.speed_line_behavior = SpeedLineType::StillLine;
        let start_pos = game.player.pos;
        game.tick_physics();
        game.update_output_buffer();
        for x in 0..3 {
            assert!(Glyph::is_braille(
                game.get_buffered_glyph(snap_to_grid(start_pos) + p(x, 0))
                    .character
            ));
        }
    }

    #[test]
    #[timeout(100)]
    fn test_player_recent_poses_starts_empty() {
        let game = set_up_player_on_platform();
        assert!(game.player.recent_kinematic_states.is_empty());
    }

    #[test]
    #[timeout(100)]
    fn test_player_recent_kinematic_states_are_saved() {
        let mut game = set_up_player_on_platform();
        let k0 = game.player.kinematic_state();
        let p1 = p(5.0, 2.0);
        let p2 = p(6.7, 3.4);
        game.tick_physics();
        game.player.pos = p1;
        game.tick_physics();
        game.player.pos = p2;
        game.tick_physics();

        assert!(game.player.recent_kinematic_states.len() == 3);
        assert!(game.player.recent_kinematic_states.get(0).unwrap().pos == p2);
        assert!(game.player.recent_kinematic_states.get(1).unwrap().pos == p1);
        assert!(game.player.recent_kinematic_states.get(2).unwrap() == &k0);
    }

    #[test]
    #[timeout(100)]
    fn test_boost_trying_to_go_faster() {
        for boost_type in BoostBehavior::iter() {
            let mut game = set_up_player_running_double_max_boost_speed_right();
            let start_vx = game.player.vel.x();
            game.player.boost_behavior = boost_type;
            game.player_dash();
            let correct_final_vx = match boost_type {
                BoostBehavior::Set => game.player.dash_vel,
                BoostBehavior::Add | BoostBehavior::AddButInstantTurnAround => {
                    start_vx + game.player.dash_vel
                }
            };
            assert!(game.player.vel.x() == correct_final_vx);
        }
    }

    #[test]
    #[timeout(100)]
    fn test_boost_trying_to_turn_around() {
        for boost_type in BoostBehavior::iter() {
            let mut game = set_up_player_running_double_max_boost_speed_right();
            game.player.desired_direction = left_i();
            let start_vx = game.player.vel.x();
            game.player.boost_behavior = boost_type;
            game.player_dash();
            let correct_final_vx = match boost_type {
                BoostBehavior::Set | BoostBehavior::AddButInstantTurnAround => {
                    -game.player.dash_vel
                }
                BoostBehavior::Add => start_vx + -game.player.dash_vel,
            };
            assert!(game.player.vel.x() == correct_final_vx);
        }
    }

    #[test]
    #[timeout(100)]
    fn test_prevent_clip_into_wall_when_colliding_with_internal_block_edge() {
        // data comes from observed bug reproduction
        let start_pos = p(35.599968, 16.5);
        let prev_pos = p(36.899967, 17.25);

        let mut game = Game::new(50, 50);
        game.player.deceleration_from_air_friction = 0.0;
        game.place_player(start_pos.x(), start_pos.y());
        game.player.vel = start_pos - prev_pos;
        game.player.desired_direction = p(-1, 0);

        game.place_block(p(34, 17), Block::Wall);
        game.place_block(p(34, 16), Block::Wall);
        game.place_block(p(34, 15), Block::Wall);

        game.num_positions_per_block_to_check_for_collisions = 0.00001;
        game.tick_physics();

        assert!(game.player.pos.x() == 35.0);
    }

    #[ignore] // removed feature
    #[test]
    #[timeout(100)]
    fn update_color_threshold_with_jump_delta_v_update() {
        let mut game = set_up_game();
        let start_thresh = game.player.speed_of_blue;
        game.set_player_jump_delta_v(game.player.jump_properties.delta_vy + 1.0);
        assert!(game.player.speed_of_blue != start_thresh);
    }

    #[ignore] // removed feature
    #[test]
    #[timeout(100)]
    fn update_color_threshold_with_speed_update() {
        let mut game = set_up_game();
        let start_thresh = game.player.speed_of_blue;
        game.set_player_max_run_speed(game.player.max_run_speed + 1.0);
        assert!(game.player.speed_of_blue != start_thresh);
    }

    #[test]
    #[timeout(100)]
    fn test_no_high_speed_color_with_normal_move_and_jump() {
        let mut game = set_up_player_on_platform();
        game.player.vel = p(game.player.max_run_speed, 0.0);
        game.player.desired_direction = p(1, 0);
        game.player_jump();
        assert!(game.get_player_color() == PLAYER_COLOR);
        assert!(game.get_player_color() != PLAYER_HIGH_SPEED_COLOR);
    }

    #[test]
    #[timeout(100)]
    fn test_no_high_speed_color_with_normal_wall_jump() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_jump_if_possible();
        assert!(game.get_player_color() != PLAYER_HIGH_SPEED_COLOR);
    }

    #[test]
    #[timeout(100)]
    fn test_toggle_bullet_time() {
        let mut game = set_up_player_on_platform();
        assert!(!game.is_bullet_time);
        game.handle_event(Event::Key(Key::Char('g')));
        assert!(game.is_bullet_time);
        game.handle_event(Event::Key(Key::Char('g')));
        assert!(!game.is_bullet_time);
    }

    #[test]
    #[timeout(100)]
    fn test_bullet_time_slows_down_motion() {
        let mut game = set_up_player_in_zero_g_frictionless_vacuum();
        game.player.vel = p(1.0, 0.0);
        let start_x = game.player.pos.x();
        game.bullet_time_factor = 0.5;
        game.toggle_bullet_time();
        game.tick_physics();
        let expected_end_x = game.player.vel.x() * game.bullet_time_factor + start_x;
        assert!(game.player.pos.x() == expected_end_x);
    }

    #[test]
    #[timeout(100)]
    fn test_bullet_time_slows_down_acceleration_from_gravity() {
        let mut game = set_up_just_player();
        game.player.jump_properties.g = 1.0;
        let start_vy = game.player.vel.y();
        game.bullet_time_factor = 0.5;
        game.toggle_bullet_time();
        game.tick_physics();
        let expected_end_vy =
            -game.get_gravitational_acceleration() * game.bullet_time_factor + start_vy;
        assert!(game.player.vel.y() == expected_end_vy);
    }

    #[test]
    #[timeout(100)]
    fn test_no_passing_max_speed_horizontally_in_midair() {
        let mut game = set_up_just_player();
        game.player.vel = p(game.player.max_run_speed, 0.0);
        game.player.desired_direction = p(1, 0);
        game.tick_physics();
        assert!(game.player.vel.x().abs() <= game.player.max_midair_move_speed);
    }

    #[test]
    #[timeout(100)]
    fn test_can_turn_around_midair_around_after_a_wall_jump() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player.deceleration_from_air_friction = 0.0;
        let start_vx = game.player.vel.x();
        game.player_jump_if_possible();
        let vx_at_start_of_jump = game.player.vel.x();
        game.tick_physics();
        let vx_one_frame_into_jump = game.player.vel.x();
        game.player.desired_direction = p(-1, 0);
        game.tick_physics();
        let end_vx = game.player.vel.x();

        assert!(start_vx == 0.0);
        assert!(vx_at_start_of_jump > 0.0);
        //assert!(vx_one_frame_into_jump == vx_at_start_of_jump);
        assert!(end_vx < vx_one_frame_into_jump);
    }

    #[test]
    #[timeout(100)]
    fn show_dash_visuals_even_if_hit_wall_in_first_frame() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(10.0, 0.0));
        game.player.desired_direction = p(0, -1);
        game.player.dash_vel = 100.0;
        game.player.speed_of_blue = 9.0;

        game.player_dash();
        game.tick_physics();
        assert!(game.get_player_color() == PLAYER_HIGH_SPEED_COLOR);
    }

    #[test]
    #[timeout(100)]
    fn do_not_hang_onto_ceiling() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(0.0, -2.0));
        game.player.desired_direction = p(0, 1);
        let start_y_pos = game.player.pos.y();
        game.tick_physics();
        assert!(game.player.pos.y() < start_y_pos);
    }

    #[test]
    #[timeout(100)]
    fn test_player_movement_compensates_for_non_square_grid() {
        let mut game = set_up_player_in_zero_g_frictionless_vacuum();
        let start_pos = game.player.pos;

        game.player.vel = p(1.0, 1.0);
        game.tick_physics();
        let movement = game.player.pos - start_pos;
        assert!(movement.x() == movement.y() * VERTICAL_GRID_STRETCH_FACTOR);
    }

    #[test]
    #[timeout(100)]
    fn test_particles__movement_compensates_for_non_square_grid() {
        let mut game = set_up_game();
        game.place_particle_with_velocity_and_lifetime(p(5.0, 5.0), p(1.0, 1.0), 500.0);

        let start_pos = game.particles[0].pos;

        game.tick_physics();
        let movement = game.particles[0].pos - start_pos;
        assert!(movement.x() == movement.y() * VERTICAL_GRID_STRETCH_FACTOR);
    }

    #[test]
    #[timeout(100)]
    fn test_no_jump_if_not_touching_floor() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(0.0, 0.1));
        assert!(!game.player_can_jump());
    }

    #[test]
    #[timeout(100)]
    fn test_player_not_standing_on_block_if_slightly_above_block() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(0.0, 0.1));
        assert!(!game.player_is_standing_on_block());
    }

    #[test]
    #[timeout(100)]
    fn test_place_particle() {
        let mut game = set_up_game();
        let pos = p(5.0, 5.0);
        game.place_particle(pos);
        assert!(game.particles.len() == 1);
        assert!(game.particles[0].pos == pos);
    }

    #[test]
    #[timeout(100)]
    fn test_draw_placed_particle() {
        let mut game = set_up_game();
        let pos = p(5.0, 5.0);
        game.place_particle(pos);
        game.update_output_buffer();
        assert!(
            *game.get_buffered_glyph(snap_to_grid(pos))
                == Glyph::world_pos_to_colored_braille_glyph(pos, ColorName::Blue)
        );
    }

    #[test]
    #[timeout(100)]
    fn test_speed_lines_are_particles() {
        let mut game = set_up_just_player();
        game.player.desired_direction = p(1, 0);
        game.player_dash();
        game.tick_physics();
        assert!(!game.particles.is_empty());
    }

    #[test]
    #[timeout(100)]
    fn test_get_particles_in_square() {
        let mut game = set_up_game();
        game.place_particle(p(0.0, 0.0));
        game.place_particle(p(5.0, 5.0));
        game.place_particle(p(5.1, 5.0));
        game.place_particle(p(6.5, 5.0));
        game.place_particle(p(6.5, 5.0));
        game.place_particle(p(6.5, 5.0));
        game.place_particle(p(6.5, 5.0));

        assert!(game.get_indexes_of_particles_in_square(p(10, 10)).len() == 0);
        assert!(game.get_indexes_of_particles_in_square(p(0, 0)).len() == 1);
        assert!(game.get_indexes_of_particles_in_square(p(5, 5)).len() == 2);
        assert!(game.get_indexes_of_particles_in_square(p(6, 5)).len() == 0);
        assert!(game.get_indexes_of_particles_in_square(p(7, 5)).len() == 4);
        assert!(game.get_indexes_of_particles_in_square(p(-7, -5)).len() == 0);
    }

    #[test]
    #[timeout(100)]
    fn test_count_braille_dots_in_square() {
        let mut game = set_up_game();
        game.place_particle(p(0.0, 0.0));
        game.place_particle(p(5.0, 5.0));
        game.place_particle(p(5.1, 5.0));

        game.place_particle(p(6.5, 5.0));
        game.place_particle(p(6.5, 5.0));
        game.place_particle(p(6.5, 5.0));
        game.place_particle(p(6.5, 5.0));

        game.update_output_buffer();

        assert!(game.count_braille_dots_in_square(p(10, 10)) == 0);
        assert!(game.count_braille_dots_in_square(p(0, 0)) == 1);
        assert!(game.count_braille_dots_in_square(p(5, 5)) == 1);
        assert!(game.count_braille_dots_in_square(p(6, 5)) == 0);
        assert!(game.count_braille_dots_in_square(p(7, 5)) == 1);
        assert!(game.count_braille_dots_in_square(p(-7, -5)) == 0);
    }

    #[test]
    #[timeout(100)]
    fn test_particle_lines_can_visually_fill_square() {
        let mut game = set_up_game();
        game.place_line_of_particles(p(5.1, 6.0), p(5.1, 4.0));
        game.place_line_of_particles(p(4.9, 6.0), p(4.9, 4.0));
        game.update_output_buffer();
        assert!(game.get_indexes_of_particles_in_square(p(5, 5)).len() == 8);
        assert!(game.count_braille_dots_in_square(p(5, 5)) == 8);
    }

    #[test]
    #[timeout(100)]
    fn test_particles__can_expire() {
        let point = p(5.0, 5.0);
        let mut game = set_up_game();
        game.place_particle_with_lifespan(point, 1.0);
        game.place_particle_with_lifespan(point, 2.0);
        assert!(game.particles.len() == 2);
        game.tick_particles();
        assert!(game.particles.len() == 1);
        game.tick_particles();
        assert!(game.particles.is_empty());
    }

    #[test]
    #[timeout(100)]
    fn test_dead_particles_are_not_drawn() {
        let point = p(5.0, 5.0);
        let mut game = set_up_game();
        game.place_particle_with_lifespan(point, 1.0);
        game.update_output_buffer();
        game.tick_particles();
        game.update_output_buffer();
        assert!(game.get_buffered_glyph(snap_to_grid(point)).character == Block::Air.character());
    }

    #[test]
    #[timeout(100)]
    fn test_particles__move_around() {
        let point = p(5.0, 5.0);
        let mut game = set_up_game();
        let mut particle = Particle::new_at(point);
        particle.random_walk_speed = DEFAULT_PARTICLE_SPEED;
        game.particles.push(particle);
        let start_pos = game.particles[0].pos;
        game.tick_particles();
        let end_pos = game.particles[0].pos;
        assert!(start_pos != end_pos);
    }

    #[test]
    #[timeout(100)]
    fn test_particles__die_when_out_of_map() {
        let mut game = set_up_game();
        let point = p(50.0, 50.0);
        game.place_particle(point);
        game.delete_out_of_bounds_particles();
        assert!(game.particles.is_empty());
    }

    #[test]
    #[timeout(100)]
    fn test_particles__random_movement_slowed_by_bullet_time() {
        let mut game1 = set_up_game();
        let mut game2 = set_up_game();
        game1.toggle_bullet_time();
        let start_pos = p(5.0, 5.0);
        game1.place_particle(start_pos);
        game1.particles[0].random_walk_speed = DEFAULT_PARTICLE_SPEED;
        game2.place_particle(start_pos);
        game2.particles[0].random_walk_speed = DEFAULT_PARTICLE_SPEED;

        game1.tick_particles();
        game2.tick_particles();

        let end_pos_1 = game1.particles[0].pos;
        let end_pos_2 = game2.particles[0].pos;
        let distorted_diff1 = end_pos_1 - start_pos;
        let distorted_diff2 = end_pos_2 - start_pos;
        let diff1 = grid_space_to_world_space(
            distorted_diff1 / game1.bullet_time_factor.sqrt(),
            VERTICAL_GRID_STRETCH_FACTOR,
        );
        let diff2 = grid_space_to_world_space(distorted_diff2, VERTICAL_GRID_STRETCH_FACTOR);

        assert!(end_pos_1 != end_pos_2);

        //dbg!(diff1, magnitude(diff1), diff2, magnitude(diff2));
        assert!(abs_diff_eq!(
            magnitude(diff1),
            magnitude(diff2),
            epsilon = 0.0001
        ));
    }

    #[test]
    #[timeout(100)]
    fn test_track_last_collision() {
        let mut game = set_up_player_in_corner_of_big_L();
        game.internal_corner_behavior = InternalCornerBehavior::StopPlayer;
        be_in_frictionless_space(&mut game);
        assert!(&game.player.last_collision.is_none());
        let collision_velocity = down_f() * 5.0;
        game.player.vel = collision_velocity;

        game.tick_physics();
        assert!(nearly_equal(
            game.ticks_since_last_player_collision().unwrap(),
            1.0
        ));
        assert!(game.player.last_collision.as_ref().unwrap().normal == p(0, 1));
        assert!(
            game.player
                .last_collision
                .as_ref()
                .unwrap()
                .collider_velocity
                == collision_velocity
        );

        game.tick_physics();
        assert!(nearly_equal(
            game.ticks_since_last_player_collision().unwrap(),
            2.0
        ));
        assert!(game.player.last_collision.as_ref().unwrap().normal == p(0, 1));
        assert!(
            game.player
                .last_collision
                .as_ref()
                .unwrap()
                .collider_velocity
                == collision_velocity
        );

        let collision_velocity = p(-10.0, 0.0);
        game.player.vel = collision_velocity;
        game.tick_physics();
        assert!(game.ticks_since_last_player_collision() == Some(1.0));
        assert!(game.player.last_collision.as_ref().unwrap().normal == p(1, 0));
        assert!(
            game.player
                .last_collision
                .as_ref()
                .unwrap()
                .collider_velocity
                == collision_velocity
        );
    }

    #[test]
    #[timeout(100)]
    fn test_last_collision_tracking_accounts_for_bullet_time() {
        let mut game = set_up_player_on_platform();
        game.player.vel = p(0.0, -1.0);
        game.tick_physics();
        let ticks_before_bullet_time = 1.0;
        assert!(nearly_equal(
            game.ticks_since_last_player_collision().unwrap(),
            ticks_before_bullet_time
        ));
        game.toggle_bullet_time();
        game.tick_physics();
        assert!(nearly_equal(
            game.ticks_since_last_player_collision().unwrap(),
            game.bullet_time_factor + ticks_before_bullet_time
        ));
        game.tick_physics();
        assert!(nearly_equal(
            game.ticks_since_last_player_collision().unwrap(),
            game.bullet_time_factor * 2.0 + ticks_before_bullet_time
        ));
    }

    #[test]
    #[timeout(100)]
    fn test_player_compresses_like_a_spring_when_colliding_at_high_speed() {
        let mut game = set_up_player_on_platform();
        game.player.vel = p(0.0, -10.0);
        game.tick_physics();
        game.tick_physics();
        assert!(game.get_player_compression_fraction() < 1.0);
        assert!(game.get_player_compression_fraction() > 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_player_compression_overrides_visuals() {
        let mut game = set_up_just_player();
        game.player.last_collision = Some(PlayerBlockCollision {
            time_in_ticks: game.time_in_ticks() - DEFAULT_TICKS_TO_MAX_COMPRESSION / 2.0,
            normal: p(0, 1),
            collider_velocity: p(0.0, -3.0),
            collider_pos: p(5.0, 5.0),
            collided_block_square: p(5, 4),
        });
        assert!(
            EIGHTH_BLOCKS_FROM_BOTTOM[1..8].contains(&game.get_compressed_player_glyph().character)
        );
    }

    #[test]
    #[timeout(100)]
    fn test_compress_horizontally_on_wall_collision() {
        let mut game = set_up_just_player();
        game.player.last_collision = Some(PlayerBlockCollision {
            time_in_ticks: game.time_in_ticks() - DEFAULT_TICKS_TO_MAX_COMPRESSION / 2.0,
            normal: p(1, 0),
            collider_velocity: p(0.0, -3.0),
            collider_pos: p(5.0, 5.0),
            collided_block_square: p(5, 4),
        }); // parts of this don't add up, but they shouldn't need to
        assert!(
            EIGHTH_BLOCKS_FROM_LEFT[1..8].contains(&game.get_compressed_player_glyph().character)
        );
    }

    #[test]
    #[timeout(100)]
    fn test_player_compression_characters_for_floor_collision_appear_in_correct_sequence() {
        let mut game = set_up_player_on_platform();
        game.player.vel = p(0.0, -10.0);
        game.tick_physics();
        assert!(game.player.last_collision.is_some());
        assert!(nearly_equal(
            game.ticks_since_last_player_collision().unwrap(),
            1.0
        ));

        let mut chars_in_compression_start = Vec::<char>::new();
        for _ in 0..DEFAULT_TICKS_TO_MAX_COMPRESSION as i32 {
            chars_in_compression_start.push(game.get_compressed_player_glyph().character);
            game.tick_physics();
        }

        let mut chars_in_compression_end = Vec::<char>::new();
        for _ in DEFAULT_TICKS_TO_MAX_COMPRESSION as i32..=DEFAULT_TICKS_TO_END_COMPRESSION as i32 {
            chars_in_compression_end.push(game.get_compressed_player_glyph().character);
            game.tick_physics();
        }
        //dbg!(&chars_in_compression_start, &chars_in_compression_end);

        assert!(*chars_in_compression_end.last().unwrap() == EIGHTH_BLOCKS_FROM_BOTTOM[8]);

        // reverse because of unicode char ordering
        chars_in_compression_start.reverse();
        assert!(chars_in_compression_start.is_sorted());
        assert!(chars_in_compression_end.is_sorted());
    }

    #[test]
    #[timeout(100)]
    fn test_leaving_floor_cancels_vertical_compression() {
        // Assignment
        let mut game = set_up_player_on_platform();
        //game.player.pos.add_assign(p(0.0, 0.1));
        game.player.vel = p(0.0, -10.0);
        game.tick_physics();
        assert!(nearly_equal(
            game.ticks_since_last_player_collision().unwrap(),
            1.0
        ));
        assert!(game.get_player_compression_fraction() < 1.0);

        // Action
        game.player_jump_if_possible();
        game.tick_physics();

        // Assertion
        assert!(game.get_player_compression_fraction() == 1.0);
    }

    #[test]
    #[timeout(100)]
    fn test_leaving_ceiling_cancels_vertical_compression() {
        // Assignment
        let mut game = set_up_player_under_platform();
        //game.player.pos.add_assign(p(0.0, 0.1));
        game.player.vel = p(0.0, 10.0);
        game.tick_physics();
        assert!(nearly_equal(
            game.ticks_since_last_player_collision().unwrap(),
            1.0
        ));

        // Action
        game.tick_physics();

        // Assertion
        assert!(game.get_player_compression_fraction() == 1.0);
    }

    #[test]
    #[timeout(100)]
    fn test_leaving_wall_cancels_horizontal_compression() {
        // Assignment
        let mut game = set_up_player_touching_wall_on_right();
        be_in_frictionless_space(&mut game);
        game.player.vel = p(10.0, 0.0);
        game.tick_physics();
        assert!(nearly_equal(
            game.ticks_since_last_player_collision().unwrap(),
            1.0
        ));
        assert!(game.get_player_compression_fraction() < 1.0);

        // Action
        game.player.vel = p(-0.1, 0.0);
        game.tick_physics();

        // Assertion
        assert!(game.get_player_compression_fraction() == 1.0);
    }

    #[test]
    #[timeout(100)]
    fn test_particles__visuals_do_not_cover_blocks() {
        let mut game = set_up_game();
        let square = p(5, 5);

        game.place_particle(floatify(square));
        game.place_block(square, Block::Wall);
        game.update_output_buffer();

        let drawn_char = game.get_buffered_glyph(square).character;

        assert!(drawn_char == Block::Wall.character());
    }

    #[test]
    #[timeout(100)]
    fn test_player_remembers_movement_normal_to_last_collision__vertical_collision() {
        let mut game = set_up_player_on_platform();
        game.player.vel = p(0.0, -1.0);
        game.tick_physics();
        assert!(nearly_equal(
            game.ticks_since_last_player_collision().unwrap(),
            1.0
        ));
        assert!(game.player.moved_normal_to_collision_since_collision == false);
        game.player_jump_if_possible();
        game.tick_physics();
        assert!(game.player.moved_normal_to_collision_since_collision == true);
    }

    #[test]
    #[timeout(100)]
    fn test_player_remembers_movement_normal_to_last_collision__horizontal_collision() {
        let mut game = set_up_player_about_to_run_into_corner_of_backward_L();
        game.internal_corner_behavior = InternalCornerBehavior::StopPlayer;
        game.tick_physics();
        assert!(game.player.last_collision.is_some());
        assert!(game.ticks_since_last_player_collision().unwrap() < 1.0);
        assert!(game.ticks_since_last_player_collision().unwrap() > 0.9);
        assert!(game.player.moved_normal_to_collision_since_collision == false);
        game.player.vel = up_f() + left_f();
        game.tick_physics();
        assert!(game.player.moved_normal_to_collision_since_collision == true);
    }

    #[test]
    #[timeout(100)]
    fn test_player_forgets_normal_movement_to_collision_upon_new_collision() {
        let mut game = set_up_player_on_platform_in_frictionless_vacuum();
        game.player.moved_normal_to_collision_since_collision = true;
        game.player.vel = p(0.0, -1.0);
        game.tick_physics();
        assert!(game.player.moved_normal_to_collision_since_collision == false);
    }

    #[test]
    #[ignore] // TODO: test compared to naive gravity
    #[timeout(100)]
    fn test_coyote_ticks_respect_bullet_time() {
        //let mut game = set_up_player_supported_by_coyote_ticks();
        //game.toggle_bullet_time();
        //game.tick_physics();
    }

    #[test]
    #[timeout(100)]
    fn test_particles__velocity() {
        let mut game = set_up_game();
        let start_pos = p(5.0, 5.0);
        let mut particle = Particle::new_at(start_pos);
        let vel = p(1.0, 0.0);
        particle.vel = vel;
        game.particles.push(particle);
        game.tick_physics();
        game.tick_physics();
        assert!(game.particles[0].pos == start_pos + vel * 2.0);
    }

    #[test]
    #[timeout(100)]
    fn test_player_compresses_when_sliding_up_wall() {
        let mut game = set_up_player_almost_touching_wall_on_right();
        game.player.desired_direction = p(1, 0);
        game.player.vel = p(10.0, 10.0);
        game.tick_physics();
        assert!(game.player.last_collision.is_some());
        game.tick_physics();
        assert!(game.get_player_compression_fraction() < 1.0);
    }

    #[ignore] // Because it is a collision
    #[test]
    #[timeout(100)]
    fn test_pushing_into_a_wall_is_not_a_collision() {
        let mut game = set_up_player_in_corner_of_backward_L();
        assert!(game.player.last_collision.is_none());
        game.player.desired_direction = p(1, 0);
        game.tick_physics();
        assert!(game.player.last_collision.is_none());
    }

    #[test]
    #[timeout(100)]
    fn test_player_exactly_touching_block_down_straight() {
        let mut game = set_up_just_player();
        let rel_wall_pos = p(0, -1);
        game.place_wall_block(snap_to_grid(game.player.pos) + rel_wall_pos);

        assert!(game.player_exactly_touching_wall_in_direction(p(0, -1)));
    }

    #[test]
    #[timeout(100)]
    fn test_player_exactly_not_touching_block_down_left_diagonal() {
        let mut game = set_up_just_player();
        let rel_wall_pos = p(-1, -1);
        game.place_wall_block(snap_to_grid(game.player.pos) + rel_wall_pos);

        assert!(!game.player_exactly_touching_wall_in_direction(p(0, -1)));
    }

    #[test]
    #[timeout(100)]
    fn test_player_exactly_touching_block_down_left_supported() {
        let mut game = set_up_just_player();
        game.player.pos.add_assign(p(-0.01, 0.0));
        let rel_wall_pos = p(-1, -1);
        game.place_wall_block(snap_to_grid(game.player.pos) + rel_wall_pos);

        assert!(game.player_exactly_touching_wall_in_direction(p(0, -1)));
    }

    #[test]
    #[timeout(100)]
    fn test_player_exactly_touching_block_on_right() {
        let mut game = set_up_just_player();
        let rel_wall_pos = p(1, 0);
        game.place_wall_block(snap_to_grid(game.player.pos) + rel_wall_pos);

        assert!(game.player_exactly_touching_wall_in_direction(p(1, 0)));
    }

    #[test]
    #[timeout(100)]
    fn test_bullet_time_slows_down_traction_acceleration() {
        let mut game1 = set_up_player_starting_to_move_right_on_platform();
        let mut game2 = set_up_player_starting_to_move_right_on_platform();
        game1.toggle_bullet_time();

        game1.tick_physics();
        game2.tick_physics();

        assert!(game1.player.vel.x() < game2.player.vel.x());
    }

    #[test]
    #[timeout(100)]
    fn test_collision_compression_starts_on_first_tick_of_collision() {
        let mut game = set_up_player_one_tick_from_platform_impact();
        assert!(game.get_player_compression_fraction() == 1.0);
        game.tick_physics();
        assert!(game.get_player_compression_fraction() < 1.0);
    }

    #[test]
    #[timeout(100)]
    fn test_moment_of_collision_has_subtick_precision() {
        let dt = 0.5;
        let mut game = set_up_player_t_ticks_from_hitting_a_block_in_space(dt);

        game.tick_physics();

        assert!(nearly_equal(
            game.ticks_since_last_player_collision().unwrap(),
            dt
        ));
    }

    #[test]
    #[timeout(100)]
    fn test_moment_of_collision_with_subtick_precision_accounts_for_non_square_grid() {
        let mut game1 = set_up_player_on_platform();
        let mut game2 = set_up_player_touching_wall_on_right();
        be_in_frictionless_space(&mut game1);
        be_in_frictionless_space(&mut game2);
        let dist_from_impact = 1.0;
        game1.player.pos.add_assign(p(0.0, dist_from_impact));
        game2.player.pos.add_assign(p(-dist_from_impact, 0.0));
        let vel_of_impact = 4.0;
        game1.player.vel = p(0.0, -vel_of_impact);
        game2.player.vel = p(vel_of_impact, 0.0);

        game1.tick_physics();
        game2.tick_physics();

        let game1_time_before_impact = 1.0 - game1.ticks_since_last_player_collision().unwrap();
        let game2_time_before_impact = 1.0 - game2.ticks_since_last_player_collision().unwrap();
        assert!(
            game1_time_before_impact == game2_time_before_impact * VERTICAL_GRID_STRETCH_FACTOR
        );
    }

    #[test]
    #[timeout(100)]
    fn test_player_supported_by_block_if_mostly_off_edge() {
        let mut game = set_up_player_on_block_more_overhanging_than_not_on_right();
        assert!(game.player_is_supported());
        assert!(game.player_is_standing_on_block());
    }

    #[ignore]
    #[test]
    #[timeout(100)]
    // The general case of this is pressing jump any time before landing causing an instant jump when possible
    fn test_allow_early_jump() {}

    #[ignore]
    #[test]
    #[timeout(100)]
    // The general case of this is allowing a single jump anytime while falling after walking (not jumping!) off a platform
    fn test_allow_late_jump() {}

    #[test]
    #[timeout(100)]
    // This should allow high skill to lead to really fast wall climbing (like in N+)
    fn test_wall_jump_adds_velocity_instead_of_sets_it() {
        let mut game1 = set_up_player_hanging_on_wall_on_left();
        let mut game2 = set_up_player_hanging_on_wall_on_left();
        let up_vel = 1.0;
        game1.player.vel = p(0.0, up_vel);
        game1.player_jump_if_possible();
        game2.player_jump_if_possible();

        assert!(game1.player.vel.y() == game2.player.vel.y() + up_vel);
    }

    #[test]
    #[timeout(100)]
    fn test_burst_speed_line_particles_have_velocity() {
        let mut game = set_up_player_just_dashed_right_in_zero_g();
        game.player.speed_line_behavior = SpeedLineType::BurstChain;
        game.tick_physics();

        assert!(!game.particles.is_empty());
        for particle in game.particles {
            assert!(magnitude(particle.vel) != 0.0);
        }
    }

    #[test]
    #[timeout(100)]
    fn test_stationary_speed_line_particles_have_no_velocity() {
        let mut game = set_up_player_just_dashed_right_in_zero_g();
        game.player.speed_line_behavior = SpeedLineType::StillLine;
        game.tick_physics();

        assert!(!game.particles.is_empty());
        for particle in game.particles {
            assert!(magnitude(particle.vel) == 0.0);
        }
    }

    #[test]
    #[timeout(100)]
    fn test_can_slow_down_midair_after_dash() {
        let mut game1 = set_up_player_just_dashed_right_in_zero_g();
        let mut game2 = set_up_player_just_dashed_right_in_zero_g();
        game2.player.desired_direction = p(-1, 0);
        game1.tick_physics();
        game2.tick_physics();

        assert!(game1.player.vel.x() > game2.player.vel.x());
    }

    #[ignore]
    #[test]
    #[timeout(100)]
    // Once we have vertical subsquare positioning up and running, a slow slide down will look cool.
    fn test_slowly_slide_down_when_grabbing_wall() {}

    #[test]
    #[timeout(100)]
    fn test_be_fastest_at_very_start_of_jump() {
        let mut game = set_up_player_running_full_speed_to_right_on_platform();
        assert!(game.player.max_run_speed >= game.player.max_midair_move_speed);
        game.player_jump();
        let vel_at_start_of_jump = game.player.vel;
        game.tick_physics();
        game.tick_physics();
        game.tick_physics();

        assert!(magnitude(game.player.vel) <= magnitude(vel_at_start_of_jump));
    }

    #[ignore]
    #[test]
    #[timeout(100)]
    fn test_fps_display() {}

    /// like a spring
    #[test]
    #[timeout(100)]
    fn test_jump_bonus_if_jump_when_coming_out_of_compression() {
        let mut game1 = set_up_player_on_platform();
        let mut game2 =
            set_up_player_halfway_through_decompressing_from_vertical_jump_on_platform();
        game1.player.enable_jump_compression_bonus = true;
        game2.player.enable_jump_compression_bonus = true;

        game1.player_jump_if_possible();
        game2.player_jump_if_possible();

        assert!(game2.player.vel.y() > game1.player.vel.y());
    }

    #[test]
    #[timeout(100)]
    fn test_full_jump_bonus_from_normal_jump_not_larger_than_normal_jump() {
        let mut game1 = set_up_player_on_platform();
        let mut game2 = set_up_player_fully_compressed_from_vertical_jump_on_platform();
        game1.player.enable_jump_compression_bonus = true;
        game2.player.enable_jump_compression_bonus = true;

        game1.player_jump_if_possible();
        game2.player_jump_if_possible();

        assert!(game2.player.vel.y() <= game1.player.vel.y() * 2.0);
    }

    #[test]
    #[timeout(100)]
    fn test_jump_bonus_only_perpendicular_to_collision__floor_collision() {
        let mut game =
            set_up_player_fully_compressed_from_vertical_jump_while_running_right_on_platform();
        game.player.enable_jump_compression_bonus = true;
        let bonus = game.jump_bonus_vel_from_compression();
        assert!(bonus.x() == 0.0);
        assert!(bonus.y() > 0.0);
    }

    #[ignore]
    #[test]
    #[timeout(100)]
    fn test_jump_bonus_NOT_only_perpendicular_to_collision__wall_collision() {
        let mut game = set_up_player_fully_compressed_from_down_leftwards_impact_on_wall();
        game.player.enable_jump_compression_bonus = true;

        let bonus = game.jump_bonus_vel_from_compression();
        assert!(bonus.x() > 0.0);
        assert!(bonus.y() > 0.0);
    }

    #[ignore]
    // just as the spring giveth, so doth the spring taketh away(-eth)
    #[test]
    #[timeout(100)]
    fn test_jump_penalty_if_jump_when_entering_compression() {}

    #[test]
    #[timeout(100)]
    fn test_perpendicular_speed_lines_move_perpendicular() {
        let dir = direction(p(1.25, 3.38)); // arbitrary
        let mut game = set_up_player_boosting_through_space_in_direction(dir);
        game.particle_rotation_speed_towards_player = 0.0;
        game.player.speed_line_behavior = SpeedLineType::PerpendicularLines;
        let v1 = game.player.vel;
        game.tick_physics();
        let v2 = game.player.vel;
        assert!(v1 == v2);
        assert!(!game.particles.is_empty());
        for p in game.particles {
            assert!(magnitude(p.vel) != 0.0);
            assert!(p.vel.dot(dir).abs() < 0.00001);
        }
    }

    #[test]
    #[timeout(100)]
    fn test_particle_wall_collision_behavior__pass_through() {
        let mut game = set_up_particle_moving_right_and_about_to_hit_wall();
        let particle_start_pos = game.particles[0].pos;
        let particle_start_vel = game.particles[0].vel;
        game.particles[0].wall_collision_behavior = ParticleWallCollisionBehavior::PassThrough;
        game.tick_particles();
        assert!(game.particles[0].pos == particle_start_pos + particle_start_vel);
        assert!(game.particles[0].vel == particle_start_vel);
    }

    #[test]
    #[timeout(100)]
    fn test_particle_wall_collision_behavior__vanish() {
        let mut game = set_up_particle_moving_right_and_about_to_hit_wall();
        game.particles[0].wall_collision_behavior = ParticleWallCollisionBehavior::Vanish;
        game.tick_particles();
        assert!(game.particles.is_empty());
    }

    #[test]
    #[timeout(100)]
    fn test_particle_wall_collision_behavior__bounce() {
        let mut game = set_up_particle_moving_right_and_about_to_hit_wall();
        //let particle_start_pos = game.particles[0].pos;
        game.particles[0].vel.set_y(-0.1);
        let particle_start_vel = game.particles[0].vel;
        game.particles[0].wall_collision_behavior = ParticleWallCollisionBehavior::Bounce;
        game.tick_particles();
        //assert!(game.particles[0].pos.x() < particle_start_pos.x());
        //assert!(game.particles[0].pos.y() == particle_start_pos.y() + particle_start_vel.y());
        assert!(game.particles[0].vel.x() == -particle_start_vel.x());
        assert!(game.particles[0].vel.y() == particle_start_vel.y());
    }

    #[test]
    #[timeout(100)]
    fn test_particles_combine_into_blocks() {
        let mut game = set_up_n_particles_about_to_move_one_square_right(
            DEFAULT_PARTICLE_DENSITY_FOR_AMALGAMATION,
        );
        let start_square = p(0, 0);
        let particle_square = p(1, 0);
        game.tick_physics();
        //dbg!(game.particles.len());
        assert!(game.particles.is_empty());
        assert!(matches!(
            game.get_block(particle_square),
            Block::ParticleAmalgam(_)
        ));
        assert!(matches!(game.get_block(start_square), Block::Air));
    }

    #[test]
    #[timeout(100)]
    fn test_particles_do_not_amalgamate_in_starting_square() {
        let mut game = set_up_30_particles_moving_slowly_right_from_origin();
        let start_square = p(0, 0);
        game.tick_physics();
        assert!(game.particles.len() == 30);
        assert!(matches!(game.get_block(start_square), Block::Air));
    }

    #[test]
    #[timeout(100)]
    fn test_particle_amalgams_absorb_particles_when_particle_lands_inside() {
        let mut game = set_up_particle_moving_right_and_about_to_hit_particle_amalgam();
        let block_square = p(5, 5);
        let block = game.get_block(block_square);
        let start_count = block.as_particle_amalgam().unwrap();
        assert!(game.particles.len() == 1);
        game.tick_physics();
        assert!(game.particles.len() == 0);
        assert!(game.get_block(block_square) == Block::ParticleAmalgam(start_count + 1));
    }

    #[test]
    #[timeout(100)]
    fn test_perpendicular_speed_lines_have_the_same_number_of_particles_in_bullet_time() {
        let dir = direction(p(1.234, 6.845)); // arbitrary
        let mut game1 = set_up_player_boosting_through_space_in_direction(dir);
        let mut game2 = set_up_player_boosting_through_space_in_direction(dir);
        game1.player.speed_line_behavior = SpeedLineType::PerpendicularLines;
        game2.player.speed_line_behavior = SpeedLineType::PerpendicularLines;

        let time_compression = 10;
        game1.bullet_time_factor = 1.0 / (time_compression as f32);
        game1.toggle_bullet_time();

        for _ in 0..time_compression {
            game1.tick_physics();
        }

        game2.tick_physics();

        assert!(game1.particles.len() == game2.particles.len());
    }

    #[test]
    #[timeout(100)]
    fn test_squarecast__hit_some_blocks() {
        let game = set_up_four_wall_blocks_at_5_and_6();
        let start_pos = p(3.0, 5.3);
        let dir = right_f();
        let collision =
            game.squarecast_for_player_collision(start_pos, start_pos + dir * 500.0, 1.0);
        assert!(collision.hit_something());
        assert!(points_nearly_equal(
            collision.collider_pos,
            p(4.0, start_pos.y())
        ));
        assert!(collision.collided_block_square.unwrap() == p(5, 5));
        assert!(collision.collision_normal.unwrap() == p(-1, 0));
    }

    #[test]
    #[timeout(100)]
    fn test_squarecast__hit_some_blocks_with_a_point() {
        let game = set_up_four_wall_blocks_at_5_and_6();
        let start_pos = p(3.0, 5.3);
        let dir = right_f();
        let collision =
            game.squarecast_for_player_collision(start_pos, start_pos + dir * 500.0, 0.0);
        assert!(collision.hit_something());
        assert!(points_nearly_equal(
            collision.collider_pos,
            p(4.5, start_pos.y())
        ));
        assert!(collision.collided_block_square.unwrap() == p(5, 5));
        assert!(collision.collision_normal.unwrap() == p(-1, 0));
    }

    #[test]
    #[timeout(100)]
    fn test_several_particles_bouncing_off_a_platform_at_various_angles_at_grid_bottom() {
        let n = 10;
        let mut game = set_up_n_particles_about_to_bounce_off_platform_at_grid_bottom(n);
        assert!(game.particles.len() == n as usize);
        let start_vels: Vec<Point<f32>> = game.particles.iter().map(|p| p.vel).collect();
        game.particle_amalgamation_density = 99999;

        game.tick_physics();

        assert!(game.particles.len() == n as usize);
        let end_vels: Vec<Point<f32>> = game.particles.iter().map(|p| p.vel).collect();

        for i in 0..game.particles.len() {
            assert!(start_vels[i].x() == end_vels[i].x());
            assert!(start_vels[i].y() == -end_vels[i].y());
            assert!(game.particles[i].pos.y() == game.particles[0].pos.y());
        }
    }

    #[test]
    #[timeout(100)]
    fn test_linecast_should_hit_very_tip_of_corner_when_next_to_other_block() {
        let game = set_up_four_wall_blocks_at_5_and_6();
        let start_pos = p(10.0, 10.0);
        let nominal_corner = p(6.5, 5.5);
        let offset_to_edge_of_exact_touch_border_corner =
            left_f() * RADIUS_OF_EXACTLY_TOUCHING_ZONE;
        let additional_offset_for_test = left_f() * RADIUS_OF_EXACTLY_TOUCHING_ZONE;
        let end_pos = nominal_corner
            + offset_to_edge_of_exact_touch_border_corner
            + additional_offset_for_test;

        let collision = game.linecast_walls_only(start_pos, end_pos);
        assert!(collision.hit_something());
        assert!(nearly_equal(collision.collider_pos.x(), 6.5));
        assert!(collision.collided_block_square.unwrap() == p(6, 6));
        assert!(collision.collision_normal.unwrap() == p(1, 0));
    }

    #[test]
    #[timeout(100)]
    fn test_linecast__hit_near_corner_at_grid_bottom() {
        let plus_center = p(7, 0);
        let game = set_up_plus_sign_wall_blocks_at_square(plus_center);
        let start_pos = floatify(plus_center) + p(-1.0, 1.0);
        let end_pos = floatify(plus_center) + p(0.0, -0.001);
        let collision = game.linecast_walls_only(start_pos, end_pos);
        assert!(collision.hit_something());
        assert!(collision.collided_block_square.unwrap() == plus_center + p(-1, 0));
        assert!(collision.collision_normal.unwrap() == p(0, 1));
    }

    #[test]
    #[timeout(100)]
    fn test_linecast_walls_only() {
        let mut game = set_up_game();
        let start_square = p(5, 5);
        let wall_square = start_square + right_i() * 2;
        game.place_wall_block(wall_square);
        game.place_block(start_square + right_i() * 1, Block::ParticleAmalgam(5));
        let collision = game.linecast_walls_only(
            floatify(start_square),
            floatify(start_square + right_i() * 5),
        );
        assert!(collision.hit_something());
        assert!(collision.collided_block_square.unwrap() == wall_square);
    }

    #[test]
    #[timeout(100)]
    fn test_particle__double_bounce_in_corner() {
        let mut game = set_up_particle_about_to_hit_concave_corner();
        let start_vel = game.particles[0].vel;

        game.tick_physics();

        let end_vel = game.particles[0].vel;
        assert!(end_vel == -start_vel);
    }

    #[test]
    #[timeout(100)]
    fn test_particle__hit_inner_corner_exactly() {
        let mut game = set_up_particle_about_to_hit_concave_corner_exactly();
        let start_vel = game.particles[0].vel;

        game.tick_physics();

        let end_vel = game.particles[0].vel;
        assert!(end_vel == -start_vel);
    }

    #[test]
    #[timeout(100)]
    fn test_particle_bounce_off_wall_at_bottom_of_grid() {
        let mut game =
            set_up_particle_moving_in_direction_about_to_hit_wall_at_square(down_f(), p(15, 0));
        game.tick_physics();
    }

    #[test]
    #[timeout(100)]
    fn test_particle_bounce_off_wall_at_top_of_grid() {
        let mut game =
            set_up_particle_moving_in_direction_about_to_hit_wall_at_square(up_f(), p(15, 29));
        game.tick_physics();
    }

    #[test]
    #[timeout(100)]
    fn test_particle_bounce_off_wall_at_right_of_grid() {
        let mut game =
            set_up_particle_moving_in_direction_about_to_hit_wall_at_square(right_f(), p(29, 15));
        game.tick_physics();
    }

    #[test]
    #[timeout(100)]
    fn test_particle_bounce_off_wall_at_left_of_grid() {
        let mut game =
            set_up_particle_moving_in_direction_about_to_hit_wall_at_square(left_f(), p(0, 15));
        game.tick_physics();
    }

    #[test]
    #[timeout(100)]
    fn test_linecast_straight_into_wall_seam() {
        let start_pos = p(0.0, 0.5);
        let end_pos = p(5.0, 0.5);
        let mut game = set_up_game();
        game.place_wall_block(p(1, 0));
        game.place_wall_block(p(1, 1));
        let collision = game.linecast_walls_only(start_pos, end_pos);
        //dbg!(&collision);
        assert!(collision.hit_something());
    }

    #[test]
    #[timeout(100)]
    fn test_occupancy_with_no_walls() {
        let game = set_up_game();
        assert!(game.get_occupancy_of_nearby_walls(p(5, 5)) == [[false; 3]; 3]);
    }

    #[test]
    #[timeout(100)]
    fn test_occupancy_with_all_walls() {
        let game = set_up_game_filled_with_walls();
        assert!(game.get_occupancy_of_nearby_walls(p(5, 5)) == [[true; 3]; 3]);
    }

    #[test]
    #[timeout(100)]
    fn test_occupancy_with_some_walls() {
        let mut game = set_up_game();
        game.place_wall_block(p(5, 5));
        game.place_wall_block(p(5, 6));
        game.place_wall_block(p(6, 5));

        // TODO: make this array formatting for convenient for xy coordinates
        assert!(
            game.get_occupancy_of_nearby_walls(p(5, 5))
                == [
                    [false, false, false],
                    [false, true, true],
                    [false, true, false],
                ]
        );
    }

    #[test]
    #[timeout(100)]
    fn test_occupancy_on_grid_edges() {
        let game = set_up_game();
        assert!(game.get_occupancy_of_nearby_walls(p(0, 0)) == [[false; 3]; 3]);
        assert!(
            game.get_occupancy_of_nearby_walls(p(game.width() as i32, game.height() as i32))
                == [[false; 3]; 3]
        );
    }

    #[test]
    #[timeout(100)]
    fn test_fall_until_particles_hit_wall_on_left() {
        let mut game = set_up_tall_drop();
        for _ in 0..100 {
            game.tick_physics();
        }
    }

    #[test]
    #[timeout(100)]
    fn test_dash_right_then_up_in_corner() {
        let mut game = set_up_player_in_bottom_right_wall_corner();
        game.player.desired_direction = p(1, 0);
        game.player_dash();
        game.tick_physics();
        game.player.desired_direction = p(0, 1);
        game.player_dash();
        game.tick_physics();
    }

    #[test]
    #[timeout(100)]
    fn test_particles_should_slowly_turn_towards_player() {
        let mut game = set_up_particle_passing_over_player_on_platform();
        let start_particle_vel = game.particles[0].vel;
        game.particle_rotation_speed_towards_player = 0.01;

        game.tick_physics();

        // less vel right
        assert!(game.particles[0].vel.x() < start_particle_vel.x());
        // more vel down
        assert!(game.particles[0].vel.y() < start_particle_vel.y());
        // same speed
        assert!(nearly_equal(
            magnitude(game.particles[0].vel),
            magnitude(start_particle_vel)
        ));
    }

    #[test]
    #[timeout(100)]
    fn test_player_does_not_get_stuck_on_nothing_on_flat_floor() {
        let N = 100;
        for i in 0..N {
            let mut game = set_up_player_starting_to_move_right_on_platform();
            let dx = i as f32 / N as f32 * 1.0;
            game.player.pos.add_assign(right_f() * dx);
            let start_pos = game.player.pos;
            game.tick_physics();
            assert!(game.player.pos != start_pos);
        }
    }

    #[test]
    #[timeout(100)]
    fn test_particle_amalgam_block_changes_appearance_with_more_particles() {
        assert!(Block::ParticleAmalgam(10).glyph() != Block::ParticleAmalgam(20).glyph());
    }

    #[test]
    #[timeout(100)]
    fn test_particle_amalgams_explode_if_too_full() {
        let mut game =
            set_up_particle_moving_right_and_about_to_hit_particle_amalgam_at_explosion_threshold();
        game.tick_physics();
        assert!(game.particles.len() as i32 == DEFAULT_PARTICLES_IN_AMALGAMATION_FOR_EXPLOSION);
        let particle_square = snap_to_grid(game.particles[0].pos);
        assert!(game.get_block(particle_square) == Block::Air);
    }

    #[test]
    #[timeout(100)]
    fn test_player_collides_with_particle_amalgams() {
        let mut game = set_up_player_on_amalgam_platform();
        let start_y = game.player.pos.y();
        game.tick_physics();
        assert!(game.player.pos.y() == start_y);
    }

    #[test]
    #[timeout(100)]
    fn test_no_particles_generated_from_only_moving_fast() {
        let mut game = set_up_player_on_platform();
        game.player.vel = up_f() * game.player.dash_vel * 10.0;
        game.tick_physics();
        assert!(game.particles.is_empty());
    }

    #[test]
    #[timeout(100)]
    fn test_verify_time_to_jump_peak_function() {
        let mut game = set_up_player_on_platform();
        be_in_vacuum(&mut game);
        game.player_jump_if_possible();

        let start_pos = game.player.pos;

        let g = game.player.jump_properties.g.abs();
        let vi = game.player.vel.y();
        let time_to_peak = JumpProperties::time_to_jump_peak(vi, g);

        assert!(vi > 0.0);
        assert!(g > 0.0);
        game.apply_physics_in_n_steps(time_to_peak, 100);
        assert!(game.player.pos.y() > start_pos.y());
        assert!(nearly_equal(game.player.vel.y(), 0.0));
        game.apply_physics_in_n_steps(time_to_peak / 2.0, 100);
        game.apply_physics_in_n_steps(time_to_peak / 2.0, 100);
        assert!(nearly_equal(game.player.pos.y(), start_pos.y()));
        game.tick_physics();
        assert!(nearly_equal(
            game.player.last_collision.unwrap().collider_velocity.y(),
            -vi
        ));
    }
    #[test]
    #[timeout(100)]
    fn test_verify_jump_duration_calculation_with_hangtime() {
        let mut game = set_up_player_on_platform();
        be_in_vacuum(&mut game);
        game.player_jump_if_possible();
        game.apply_physics_in_n_steps(game.player.jump_properties.duration, 100);
        assert!(game.ticks_since_last_player_collision().unwrap() < 1.0);
    }

    #[test]
    #[timeout(100)]
    fn test_verify_calc_jump_height() {
        let mut game = set_up_player_on_platform();
        be_in_vacuum(&mut game);
        game.player_jump_if_possible();

        let y0 = game.player.pos.y();

        let g = game.player.jump_properties.g.abs();
        let vi = game.player.vel.y();
        let time_to_peak = JumpProperties::time_to_jump_peak(vi, g);
        let jump_peak_y =
            y0 + JumpProperties::height_from_jump_speed_and_g(vi, g) / VERTICAL_GRID_STRETCH_FACTOR;

        assert!(vi > 0.0);
        assert!(g > 0.0);
        game.apply_physics_in_n_steps(time_to_peak, 100);
        assert!(nearly_equal(game.player.vel.y(), 0.0));
        assert!(nearly_equal(game.player.pos.y(), jump_peak_y));
        game.apply_physics_in_n_steps(time_to_peak / 2.0, 100);
        game.apply_physics_in_n_steps(time_to_peak / 2.0, 100);
        assert!(nearly_equal(game.player.pos.y(), y0));
    }

    #[test]
    fn test_validate_jump_utility_functions_against_kinematic_state_extrapolation() {
        let yi = 5.0;
        let vi = 1.0;
        let g = 0.1;
        let state = KinematicState {
            pos: up_f() * yi,
            vel: up_f() * vi,
            accel: down_f() * g,
        };

        let expected_time_to_peak = JumpProperties::time_to_jump_peak(vi, g);
        assert!(nearly_equal(
            expected_time_to_peak,
            state.dt_to_slowest_point()
        ));
        let expected_height = JumpProperties::height_from_jump_speed_and_g(vi, g);
        assert!(nearly_equal(
            expected_height,
            state
                .extrapolated_delta(state.dt_to_slowest_point())
                .pos
                .y()
        ));
    }

    #[test]
    #[timeout(100)]
    fn test_can_kill_player_off_screen() {
        let mut game = set_up_just_player();
        game.player.pos = p(game.width() as f32 * 5.0, game.height() as f32 * 5.0);
        game.kill_player();
    }

    #[test]
    #[timeout(100)]
    fn test_do_not_warp_at_peak_of_jump() {
        let mut game = set_up_player_about_to_reach_peak_of_jump();
        game.player.jump_properties.g = 0.1;
        let start_pos = game.player.pos;
        game.tick_physics();
        let distance_from_start = magnitude(game.player.pos - start_pos);
        assert!(distance_from_start < 0.5);
    }

    #[test]
    #[timeout(100)]
    fn test_player_wants_to_stop_by_aiming_at_floor() {
        let mut game = set_up_player_on_platform();
        game.player.desired_direction = down_i();
        assert!(game.player_wants_to_come_to_a_full_stop());
    }

    #[test]
    #[timeout(100)]
    fn test_player_does_not_want_to_stop_if_aiming_forwards() {
        let mut game = set_up_player_running_full_speed_to_right_on_platform();
        game.player.desired_direction = right_i();
        assert!(!game.player_wants_to_come_to_a_full_stop());
    }

    #[test]
    #[timeout(100)]
    fn test_player_does_not_want_to_stop_if_aiming_backwards() {
        let mut game = set_up_player_running_full_speed_to_right_on_platform();
        game.player.desired_direction = left_i();
        assert!(!game.player_wants_to_come_to_a_full_stop());
    }

    #[test]
    #[timeout(100)]
    fn test_player_wants_to_stop_when_grabbing_wall() {
        let game = set_up_player_hanging_on_wall_on_left();
        assert!(game.player_wants_to_come_to_a_full_stop());
    }

    #[test]
    #[timeout(100)]
    fn test_player_does_not_want_to_stop_when_midair() {
        let game = set_up_just_player();
        assert!(!game.player_wants_to_come_to_a_full_stop());
    }

    #[test]
    #[timeout(100)]
    fn test_player_cannot_fly() {
        let mut game = set_up_just_player();
        game.player.desired_direction = up_i();
        game.tick_physics();
        assert!(game.player.vel.y() < 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_ground_friction_is_left_right_symmetric() {
        let mut game1 = set_up_player_running_double_max_speed_left();
        let mut game2 = set_up_player_running_double_max_run_speed_right();

        assert!(game1.player.vel.x().abs() == game2.player.vel.x().abs());
        game1.tick_physics();
        game2.tick_physics();
        assert!(game1.player.vel.x().abs() == game2.player.vel.x().abs());
    }

    #[test]
    #[timeout(100)]
    fn test_grab_wall_after_impact_with_wall() {
        let mut game = set_up_player_about_to_hit_wall_on_right_midair();
        game.tick_physics();
        assert!(game.player_is_grabbing_wall());
        game.tick_physics();
        assert!(game.player_is_grabbing_wall());
        assert!(game.player.vel.y() == 0.0);
        assert!(game.player.accel == zero_f());
    }

    #[test]
    #[timeout(100)]
    fn test_boost_particles_interpolate_starting_velocities() {
        let mut game = set_up_just_player();
        game.player.desired_direction = right_i();
        game.player_dash();
        game.tick_physics();
        assert!(game.particles.len() > 5); // approx enough particles for test to be meaningful
        for i in 2..game.particles.len() {
            //dbg!(game.particles[i].vel);
            assert!(game.particles[i].vel != game.particles[i - 1].vel);
            assert!(game.particles[i].vel != game.particles[i - 2].vel);
        }
    }

    #[test]
    #[timeout(100)]
    fn test_turret_laser_direction_starts_non_zero() {
        let turret = Turret::new();
        assert!(turret.laser_direction != zero_f());
    }

    #[test]
    #[timeout(100)]
    fn test_place_turret() {
        let mut game = set_up_game();
        let turret_square = p(5, 5);
        game.place_turret(turret_square);
        assert!(game.turrets.len() == 1);
        assert!(game.turrets[0].square == turret_square);
        assert!(game.turrets[0].laser_direction == up_f());
        assert!(game.get_block(turret_square) == Block::Turret);
    }

    #[test]
    #[timeout(100)]
    fn test_turret_is_drawn() {
        let mut game = set_up_turret_facing_up();
        game.update_output_buffer();
        assert!(game.get_buffered_glyph(game.turrets[0].square) == &Block::Turret.glyph());
    }

    #[test]
    #[timeout(100)]
    fn test_turret_fires_laser() {
        let mut game = set_up_turret_facing_up();
        game.tick_physics();
        assert!(game.turrets[0].laser_firing_result.is_some());
    }

    #[test]
    #[timeout(100)]
    fn test_turret_laser_is_visible_red_braille() {
        let mut game = set_up_turret_facing_up();
        game.tick_physics();
        game.update_output_buffer();
        let square_that_should_be_braille = game.turrets[0].square + up_i();
        let glyph = game.get_buffered_glyph(square_that_should_be_braille);
        assert!(Glyph::is_braille(glyph.character));
        assert!(glyph.fg_color == ColorName::Red);
    }

    #[test]
    #[timeout(100)]
    fn test_turret_lasers_rotate() {
        let mut game = set_up_turret_facing_up();
        let start_laser_dir = game.turrets[0].laser_direction;
        game.tick_physics();
        let end_laser_dir = game.turrets[0].laser_direction;
        assert!(start_laser_dir != end_laser_dir);
    }

    #[test]
    #[timeout(100)]
    fn test_laser_hits_particle() {
        let mut game = set_up_one_stationary_particle();
        let particle_pos = game.particles[0].pos;

        assert!(game.particles.len() == 1);

        game.update_particle_location_map();
        let laser_result = game.fire_laser(
            particle_pos + left_f() * 3.0,
            particle_pos + right_f() * 3.0,
        );

        assert!(game.particles.len() == 1);
        assert!(laser_result.hit_something());
        assert!(laser_result.collided_particle_index == Some(0));
    }

    #[test]
    #[timeout(100)]
    fn test_turret_laser_destroys_particle() {
        let mut game = set_up_turret_aiming_at_stationary_particle();

        assert!(game.particles.len() == 1);

        game.tick_physics();

        assert!(game.particles.len() == 0);
    }

    #[test]
    #[timeout(100)]
    fn test_step_foe_exists() {
        let mut game = set_up_single_step_foe();

        assert!(!game.step_foes.is_empty());
    }

    #[ignore] // maybe later
    #[test]
    #[timeout(100)]
    fn test_turrets_and_step_foes_take_up_two_squares() {}

    #[test]
    #[timeout(100)]
    fn test_player_dies_if_they_are_killed() {
        let mut game = set_up_just_player();
        assert!(game.player.alive);
        game.kill_player();
        assert!(!game.player.alive);
    }

    #[test]
    #[timeout(100)]
    fn test_particles_on_player_death() {
        let mut game = set_up_just_player();
        assert!(game.particles.is_empty());
        game.kill_player();
        assert!(game.particles.len() > 30);
    }

    #[test]
    #[timeout(100)]
    fn test_k_button_kills_player() {
        let mut game = set_up_just_player();
        assert!(game.player.alive);
        game.handle_event(Event::Key(Key::Char('k')));
        assert!(!game.player.alive);
    }

    #[test]
    #[timeout(100)]
    fn test_particle_amalgamation_gets_all_particles_in_square() {
        let mut game = set_up_particles_about_to_combine();
        game.place_n_particles(5, game.particles[0].pos);
        game.combine_dense_particles();

        assert!(game.particles.is_empty());
    }

    #[test]
    #[timeout(100)]
    fn test_existing_particle_amalgams_do_not_grab_new_particles_on_same_square() {
        let mut game = set_up_particles_about_to_combine();
        let particle_pos = game.particles[0].pos;
        game.combine_dense_particles();
        assert!(game.particles.is_empty());
        game.place_n_particles(5, particle_pos);
        game.combine_dense_particles();
        assert!(!game.particles.is_empty());
    }

    #[test]
    #[timeout(100)]
    fn test_lasers_decay_particle_amalgams() {
        let mut game = set_up_turret_aiming_at_particle_amalgam();
        let the_turret = &game.turrets[0];
        let amalgam_square = the_turret.square + up_i() * 3;
        assert!(matches!(
            game.get_block(amalgam_square),
            Block::ParticleAmalgam(_)
        ));
        assert!(game.particles.is_empty());
        game.tick_physics();
        let laser_result = game.turrets[0].laser_firing_result.unwrap();
        assert!(laser_result.hit_something());
        assert!(laser_result.hit_block());
        assert!(laser_result.collided_block_square.unwrap() == amalgam_square);
        assert!(game.particles.len() == 1);
        assert!(snap_to_grid(game.particles[0].pos) == amalgam_square);
    }

    #[test]
    #[timeout(100)]
    fn test_turret_does_not_laser_itself() {
        let mut game = set_up_turret_facing_direction(down_f());
        game.tick_physics();
        let maybe_hit_square = game.turrets[0]
            .laser_firing_result
            .unwrap()
            .collided_block_square;
        assert!(maybe_hit_square.is_none());
    }

    #[test]
    #[timeout(100)]
    fn test_internal_wall_corner_momentum_interaction__right_to_up() {
        for internal_corner_behavior in InternalCornerBehavior::iter() {
            let mut game = set_up_player_about_to_run_into_corner_of_backward_L();
            game.internal_corner_behavior = internal_corner_behavior;
            game.tick_physics();
            match game.internal_corner_behavior {
                InternalCornerBehavior::StopPlayer => assert!(game.player.vel == zero_f()),
                InternalCornerBehavior::RedirectPlayer => {
                    assert!(game.player.vel.x() == 0.0);
                    assert!(game.player.vel.y() > 0.0);
                }
            }
        }
    }

    #[test]
    #[timeout(100)]
    fn test_internal_wall_corner_momentum_interaction__down_to_right() {
        for internal_corner_behavior in InternalCornerBehavior::iter() {
            let mut game = set_up_player_in_corner_of_big_L();
            game.internal_corner_behavior = internal_corner_behavior;
            let square_pos = floatify(game.player_square());
            game.player.pos.add_assign(up_f() * 0.01);
            game.player.vel = down_f() * 20.0;
            game.tick_physics();
            assert!(game.player.pos.y() == square_pos.y());
            match game.internal_corner_behavior {
                InternalCornerBehavior::StopPlayer => {
                    assert!(game.player.vel == zero_f());
                    assert!(game.player.pos.x() == square_pos.x());
                }
                InternalCornerBehavior::RedirectPlayer => {
                    assert!(game.player.pos.x() > square_pos.x());
                    assert!(game.player.vel.y() == 0.0);
                    assert!(game.player.vel.x() > 0.0);
                }
            }
        }
    }

    #[ignore] // TODO
    #[test]
    #[timeout(100)]
    fn test_turrets_and_step_foes_flash_red_when_hit_by_particle() {}

    #[ignore] // TODO
    #[test]
    #[timeout(100)]
    fn test_particle_amalgams_attract_particles() {}

    #[ignore] // TODO
    #[test]
    #[timeout(100)]
    fn test_turrets_attract_particles() {}

    #[ignore] // TODO
    #[test]
    #[timeout(100)]
    fn test_step_foes_move_towards_player() {}

    #[test]
    #[timeout(100)]
    fn test_getting_block_adjacency() {
        let game = set_up_player_in_corner_of_big_L();
        let player_square = game.player_square();
        let adjacency = game.get_square_adjacency(player_square).unwrap();
        let correct_adjacency = visible_xy_to_actual_xy([
            [true, false, false],
            [true, false, false],
            [true, true, true],
        ]);
        assert!(adjacency == correct_adjacency);
    }

    #[test]
    #[timeout(100)]
    fn test_player_only_needs_to_be_next_to_a_wall_to_count_as_running_up_it() {
        let mut game = set_up_player_in_corner_of_big_L();
        game.player_jump_if_possible();
        assert!(game.player_is_running_up_wall());
    }

    #[test]
    #[ignore] // TODO: test compared to naive gravity
    #[timeout(100)]
    fn test_jumps_have_hangtime_at_peak() {
        let mut game = set_up_player_about_to_reach_peak_of_jump();
        let naive_jump_duration = game.player.jump_properties.duration;
        assert!(game.player.vel.y() > 0.0);
        game.tick_physics();
        assert!(nearly_equal(game.player.vel.y(), 0.0));
    }
}
