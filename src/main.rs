#![allow(non_snake_case)]
#![feature(is_sorted)]
mod glyph;
mod utility;

extern crate geo;
extern crate line_drawing;
extern crate num;
extern crate std;
extern crate termion;
#[macro_use]
extern crate approx;

use ntest::timeout;

// use assert2::{assert, check};
use enum_as_inner::EnumAsInner;
use geo::algorithm::euclidean_distance::EuclideanDistance;
use geo::Point;
use num::traits::FloatConst;
use std::char;
use std::cmp::min;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::io::{stdin, stdout, Write};
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, Instant};
use termion::event::{Event, Key, MouseButton, MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::IntoRawMode;

use glyph::*;
use utility::*;

type fpoint = Point<f32>;
type ipoint = Point<i32>;

// const player_jump_height: i32 = 3;
// const player_jump_hang_frames: i32 = 4;
const MAX_FPS: i32 = 60; // frames per second
const IDEAL_FRAME_DURATION_MS: u128 = (1000.0 / MAX_FPS as f32) as u128;
const PLAYER_COLOR: ColorName = ColorName::Red;
const PLAYER_HIGH_SPEED_COLOR: ColorName = ColorName::Blue;
const NUM_SAVED_PLAYER_POSES: i32 = 10;
const NUM_POSITIONS_TO_CHECK_PER_BLOCK_FOR_COLLISIONS: f32 = 8.0;

// a block every two ticks
const DEFAULT_PLAYER_COYOTE_TIME_DURATION_S: f32 = 0.1;
const DEFAULT_PLAYER_MAX_COYOTE_TIME: f32 =
    (DEFAULT_PLAYER_COYOTE_TIME_DURATION_S * MAX_FPS as f32) + 1.0;

const DEFAULT_PLAYER_JUMP_DELTA_V: f32 = 1.0;

const VERTICAL_STRETCH_FACTOR: f32 = 2.0; // because the grid is not really square

const DEFAULT_PLAYER_ACCELERATION_FROM_GRAVITY: f32 = 0.1;
const DEFAULT_PLAYER_ACCELERATION_FROM_FLOOR_TRACTION: f32 = 1.0;
const DEFAULT_PLAYER_ACCELERATION_FROM_AIR_TRACTION: f32 =
    DEFAULT_PLAYER_ACCELERATION_FROM_FLOOR_TRACTION;
const DEFAULT_PLAYER_GROUND_FRICTION_DECELERATION: f32 =
    DEFAULT_PLAYER_ACCELERATION_FROM_FLOOR_TRACTION / 5.0;
const DEFAULT_PLAYER_MAX_RUN_SPEED: f32 = 0.5;
const DEFAULT_PLAYER_GROUND_FRICTION_START_SPEED: f32 = 0.7;
const DEFAULT_PLAYER_DASH_SPEED: f32 = DEFAULT_PLAYER_MAX_RUN_SPEED * 3.0;
const DEFAULT_PLAYER_AIR_FRICTION_DECELERATION: f32 = 0.0;
const DEFAULT_PLAYER_MIDAIR_MAX_MOVE_SPEED: f32 = DEFAULT_PLAYER_MAX_RUN_SPEED;
const DEFAULT_PLAYER_AIR_FRICTION_START_SPEED: f32 = DEFAULT_PLAYER_DASH_SPEED;

const DEFAULT_PARTICLE_LIFETIME_IN_SECONDS: f32 = 0.5;
const DEFAULT_PARTICLE_LIFETIME_IN_TICKS: f32 =
    DEFAULT_PARTICLE_LIFETIME_IN_SECONDS * MAX_FPS as f32;
const DEFAULT_PARTICLE_STEP_PER_TICK: f32 = 0.1;
const DEFAULT_MAX_COMPRESSION: f32 = 0.2;
const DEFAULT_TICKS_TO_MAX_COMPRESSION: f32 = 5.0;
const DEFAULT_TICKS_TO_END_COMPRESSION: f32 = 10.0;

const DEFAULT_PARTICLE_AMALGAMATION_DENSITY: i32 = 10;

const RADIUS_OF_EXACTLY_TOUCHING_ZONE: f32 = 0.000001;

pub type AdjacentOccupancyMask = [[bool; 3]; 3];

// These have no positional information
#[derive(Copy, Clone, PartialEq, Eq, Debug, EnumAsInner)]
enum Block {
    Air,
    Wall,
    Brick,
    ParticleAmalgam(i32),
}

impl Block {
    fn character(&self) -> char {
        match self {
            Block::Air => ' ',
            Block::Brick => '▪',
            Block::Wall => '█',
            Block::ParticleAmalgam(_) => '▒',
        }
    }
    fn color(&self) -> ColorName {
        match self {
            Block::Air => ColorName::Black,
            Block::Wall => ColorName::White,
            Block::Brick => ColorName::White,
            Block::ParticleAmalgam(_) => ColorName::Blue,
        }
    }

    fn subject_to_block_gravity(&self) -> bool {
        match self {
            Block::Air | Block::Wall | Block::ParticleAmalgam(_) => false,
            _ => true,
        }
    }
    #[allow(dead_code)]
    fn wall_grabbable(&self) -> bool {
        match self {
            Block::Air => false,
            _ => true,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
enum ParticleWallCollisionBehavior {
    PassThrough,
    Vanish,
    Bounce,
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
}

#[derive(Debug)]
struct PlayerBlockCollision {
    time_in_ticks: f32,
    normal: Point<i32>,
    collider_velocity: Point<f32>,
    #[allow(dead_code)]
    collider_pos: Point<f32>,
    #[allow(dead_code)]
    collided_block_square: Point<i32>,
}

#[derive(PartialEq, Debug)]
enum SpeedLineType {
    StillLine,
    BurstChain,
    #[allow(dead_code)]
    BurstOnDash,
    PerpendicularLines,
}

struct Player {
    alive: bool,
    pos: Point<f32>,
    recent_poses: VecDeque<Point<f32>>,
    max_run_speed: f32,
    ground_friction_start_speed: f32,
    air_friction_start_speed: f32,
    max_midair_move_speed: f32,
    color_change_speed_threshold: f32,
    vel: Point<f32>,
    desired_direction: Point<i32>,
    jump_delta_v: f32,
    acceleration_from_gravity: f32,
    acceleration_from_floor_traction: f32,
    acceleration_from_air_traction: f32,
    deceleration_from_air_friction: f32,
    deceleration_from_ground_friction: f32,
    remaining_coyote_time: f32,
    max_coyote_time: f32,
    dash_vel: f32,
    dash_adds_to_vel: bool,
    last_collision: Option<PlayerBlockCollision>,
    moved_normal_to_collision_since_collision: bool,
    speed_line_lifetime_in_ticks: f32,
    speed_line_behavior: SpeedLineType,
}

impl Player {
    fn new() -> Player {
        Player {
            alive: false,
            pos: p(0.0, 0.0),
            recent_poses: VecDeque::<Point<f32>>::new(),
            max_run_speed: DEFAULT_PLAYER_MAX_RUN_SPEED,
            ground_friction_start_speed: DEFAULT_PLAYER_GROUND_FRICTION_START_SPEED,
            air_friction_start_speed: DEFAULT_PLAYER_AIR_FRICTION_START_SPEED,
            max_midair_move_speed: DEFAULT_PLAYER_MIDAIR_MAX_MOVE_SPEED,
            color_change_speed_threshold: magnitude(p(
                DEFAULT_PLAYER_MAX_RUN_SPEED,
                DEFAULT_PLAYER_JUMP_DELTA_V,
            )),
            vel: Point::<f32>::new(0.0, 0.0),
            desired_direction: p(0, 0),
            jump_delta_v: DEFAULT_PLAYER_JUMP_DELTA_V,
            acceleration_from_gravity: DEFAULT_PLAYER_ACCELERATION_FROM_GRAVITY,
            acceleration_from_floor_traction: DEFAULT_PLAYER_ACCELERATION_FROM_FLOOR_TRACTION,
            acceleration_from_air_traction: DEFAULT_PLAYER_ACCELERATION_FROM_AIR_TRACTION,
            deceleration_from_air_friction: DEFAULT_PLAYER_AIR_FRICTION_DECELERATION,
            deceleration_from_ground_friction: DEFAULT_PLAYER_GROUND_FRICTION_DECELERATION,
            remaining_coyote_time: DEFAULT_PLAYER_MAX_COYOTE_TIME,
            max_coyote_time: DEFAULT_PLAYER_MAX_COYOTE_TIME,
            dash_vel: DEFAULT_PLAYER_DASH_SPEED,
            dash_adds_to_vel: false,
            last_collision: None,
            moved_normal_to_collision_since_collision: false,
            speed_line_lifetime_in_ticks: 0.5 * MAX_FPS as f32,
            speed_line_behavior: SpeedLineType::PerpendicularLines,
        }
    }
}

struct Game {
    time_from_start_in_ticks: f32,
    grid: Vec<Vec<Block>>,             // (x,y), left to right, top to bottom
    output_buffer: Vec<Vec<Glyph>>,    // (x,y), left to right, top to bottom
    output_on_screen: Vec<Vec<Glyph>>, // (x,y), left to right, top to bottom
    particles: Vec<Particle>,
    terminal_size: (u16, u16),  // (width, height)
    prev_mouse_pos: (i32, i32), // where mouse was last frame (if pressed)
    running: bool,              // set false to quit
    selected_block: Block,      // What the mouse places
    num_positions_per_block_to_check_for_collisions: f32,
    is_bullet_time: bool,
    bullet_time_factor: f32,
    player: Player,
    particle_amalgamation_density: i32,
}

impl Game {
    fn new(width: u16, height: u16) -> Game {
        Game {
            time_from_start_in_ticks: 0.0,
            grid: vec![vec![Block::Air; height as usize]; width as usize],
            output_buffer: vec![vec![Glyph::from_char(' '); height as usize]; width as usize],
            output_on_screen: vec![vec![Glyph::from_char('x'); height as usize]; width as usize],
            particles: Vec::<Particle>::new(),
            terminal_size: (width, height),
            prev_mouse_pos: (1, 1),
            running: true,
            selected_block: Block::Wall,
            num_positions_per_block_to_check_for_collisions:
                NUM_POSITIONS_TO_CHECK_PER_BLOCK_FOR_COLLISIONS,
            is_bullet_time: false,
            bullet_time_factor: 0.1,
            player: Player::new(),
            particle_amalgamation_density: DEFAULT_PARTICLE_AMALGAMATION_DENSITY,
        }
    }

    fn time_in_ticks(&self) -> f32 {
        return self.time_from_start_in_ticks;
    }

    fn time_since(&self, t: f32) -> f32 {
        return self.time_in_ticks() - t;
    }

    fn time_since_last_player_collision(&self) -> Option<f32> {
        if self.player.last_collision.is_some() {
            Some(self.time_since(self.player.last_collision.as_ref().unwrap().time_in_ticks))
        } else {
            None
        }
    }

    fn set_player_jump_delta_v(&mut self, delta_v: f32) {
        self.player.jump_delta_v = delta_v;
        self.update_player_color_change_speed_thresh();
    }

    fn set_player_max_run_speed(&mut self, speed: f32) {
        self.player.max_run_speed = speed;
        self.update_player_color_change_speed_thresh();
    }

    fn update_player_color_change_speed_thresh(&mut self) {
        self.player.color_change_speed_threshold =
            magnitude(p(self.player.max_run_speed, self.player.jump_delta_v));
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
        return self.grid[square.x() as usize][square.y() as usize];
    }
    fn try_get_block(&self, square: Point<i32>) -> Option<Block> {
        return if self.in_world(square) {
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

    fn place_block(&mut self, pos: Point<i32>, block: Block) {
        if !self.in_world(pos) {
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
        return if self.in_world(square) {
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
                    if !self.in_world(grid_square) {
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
        self.player.remaining_coyote_time = 0.0;
    }
    fn player_jump(&mut self) {
        if self.player_is_grabbing_wall() || self.player_is_running_up_wall() {
            let wall_direction = sign(self.player.desired_direction);
            self.player
                .vel
                .add_assign(floatify(-wall_direction) * self.player.max_run_speed);
            self.player.desired_direction = -wall_direction;
        }
        self.player.vel.add_assign(p(0.0, self.player.jump_delta_v));
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
            if self.player.dash_adds_to_vel {
                self.player.vel.add_assign(dash_vel);
            } else {
                self.player.vel = dash_vel;
            }
        }
    }
    fn toggle_bullet_time(&mut self) {
        self.is_bullet_time = !self.is_bullet_time;
    }

    fn handle_event(&mut self, evt: termion::event::Event) {
        match evt {
            Event::Key(ke) => match ke {
                Key::Char('q') => self.running = false,
                Key::Char('1') => self.selected_block = Block::Air,
                Key::Char('2') => self.selected_block = Block::Wall,
                Key::Char('3') => self.selected_block = Block::Brick,
                Key::Char('c') => self.clear(),
                Key::Char('r') => self.place_player(
                    self.terminal_size.0 as f32 / 2.0,
                    self.terminal_size.1 as f32 / 2.0,
                ),
                Key::Char(' ') => self.player_jump_if_possible(),
                Key::Char('f') => self.player_dash(),
                Key::Char('g') => self.toggle_bullet_time(),
                Key::Char('w') | Key::Up => self.player.desired_direction = p(0, 1),
                Key::Char('a') | Key::Left => self.player.desired_direction = p(-1, 0),
                Key::Char('s') | Key::Down => self.player.desired_direction = p(0, -1),
                Key::Char('d') | Key::Right => self.player.desired_direction = p(1, 0),
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
            self.apply_forces_to_player(dt_in_ticks);
            self.apply_player_kinematics(dt_in_ticks);
            if self.player_is_officially_fast() {
                self.generate_speed_particles();
            }
        }
        self.apply_particle_physics(dt_in_ticks);
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

    fn tick_particles(&mut self) {
        self.apply_particle_physics(self.get_time_factor());
    }

    fn apply_particle_physics(&mut self, dt_in_ticks: f32) {
        self.apply_particle_lifetimes(dt_in_ticks);

        self.apply_particle_velocities(dt_in_ticks);

        self.delete_out_of_bounds_particles();

        self.combine_dense_particles();
    }

    fn get_particle_histogram(&self) -> HashMap<Point<i32>, Vec<usize>> {
        let mut histogram = HashMap::<Point<i32>, Vec<usize>>::new();
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
        histogram
    }

    fn combine_dense_particles(&mut self) {
        let mut particle_indexes_to_delete = vec![];
        for (square, indexes_of_particles_in_square) in self.get_particle_histogram() {
            let mut indexes_of_particles_that_did_not_start_here: Vec<usize> =
                indexes_of_particles_in_square
                    .iter()
                    .filter(|&&index| snap_to_grid(self.particles[index].start_pos) != square)
                    .cloned()
                    .collect();
            let block = self.get_block(square);
            if indexes_of_particles_that_did_not_start_here.len()
                > self.particle_amalgamation_density as usize
                || (!indexes_of_particles_that_did_not_start_here.is_empty()
                    && matches!(block, Block::ParticleAmalgam(_)))
            {
                let existing_count = if let Block::ParticleAmalgam(count) = block {
                    count
                } else {
                    0
                };
                self.set_block(
                    square,
                    Block::ParticleAmalgam(
                        indexes_of_particles_that_did_not_start_here.len() as i32 + existing_count,
                    ),
                );
                particle_indexes_to_delete
                    .append(&mut indexes_of_particles_that_did_not_start_here);
            }
        }
        self.delete_particles_at_indexes(particle_indexes_to_delete);
    }

    fn delete_particles_at_indexes(&mut self, mut indexes: Vec<usize>) {
        indexes.sort_unstable();
        indexes.dedup();
        indexes.reverse();

        for i in indexes {
            self.particles.remove(i);
        }
    }

    fn apply_particle_velocities(&mut self, dt_in_ticks: f32) {
        //dbg!(self.time_from_start_in_ticks);
        let mut particle_indexes_to_delete = vec![];
        for i in 0..self.particles.len() {
            let particle = self.particles[i].clone();

            let mut step = world_space_to_grid_space(
                particle.vel * dt_in_ticks
                    + direction(random_direction())
                        * particle.random_walk_speed
                        * dt_in_ticks.sqrt(),
                VERTICAL_STRETCH_FACTOR,
            );
            //dbg!(&step);
            let mut start_pos = particle.pos;
            let mut end_pos = start_pos + step;

            if particle.wall_collision_behavior != ParticleWallCollisionBehavior::PassThrough {
                while magnitude(step) > 0.00001 {
                    if let Some(collision) = self.linecast(start_pos, end_pos) {
                        if particle.wall_collision_behavior == ParticleWallCollisionBehavior::Bounce
                        {
                            //dbg!(&particle, &collision);
                            let new_start = collision.collider_pos;
                            let step_taken = new_start - start_pos;
                            step.add_assign(-step_taken);
                            let vel = self.particles[i].vel;
                            if collision.normal.x() != 0 {
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
                let particle_ended_inside_square = self.get_block(end_square) == Block::Wall
                    && point_inside_square(end_pos, end_square);
                if particle_ended_inside_square
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
            .map(|particle| self.in_world(snap_to_grid(particle.pos)))
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
        if self.player_is_officially_fast() {
            PLAYER_HIGH_SPEED_COLOR
        } else {
            PLAYER_COLOR
        }
    }

    fn get_player_compression_fraction(&self) -> f32 {
        if self.player.moved_normal_to_collision_since_collision {
            return 1.0;
        }

        if let Some(ticks_from_collision) = self.time_since_last_player_collision() {
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

    fn player_is_officially_fast(&self) -> bool {
        let mut inferred_speed = 0.0;
        if let Some(last_pos) = self.player.recent_poses.get(0) {
            inferred_speed = magnitude(self.player.pos - *last_pos);
        }
        let actual_speed = magnitude(self.player.vel);
        actual_speed > self.player.color_change_speed_threshold
            || inferred_speed > self.player.color_change_speed_threshold
    }

    fn update_output_buffer(&mut self) {
        self.fill_output_buffer_with_black();
        self.draw_particles();
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

                    if self.in_world(p(x, y)) {
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

    fn fill_output_buffer_with_black(&mut self) {
        let width = self.grid.len();
        let height = self.grid[0].len();
        for x in 0..width {
            for y in 0..height {
                self.output_buffer[x][y] = Glyph::from_char(Block::Air.character());
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
        if let Some(&last_pos) = self.player.recent_poses.get(0) {
            match &self.player.speed_line_behavior {
                SpeedLineType::StillLine => {
                    self.place_static_speed_lines(last_pos, self.player.pos)
                }
                SpeedLineType::PerpendicularLines => self.place_perpendicular_moving_speed_lines(
                    last_pos,
                    self.player.pos,
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
        start_pos: Point<f32>,
        end_pos: Point<f32>,
        start_time: f32,
        end_time: f32,
    ) {
        let particle_speed = 0.3;
        // Note: Due to non-square nature of the grid, player velocity may not be parallel to displacement
        let particle_vel = direction(self.player.vel) * particle_speed;
        let particles_per_block = 2.0;
        let time_frequency_of_speed_particles = particles_per_block * magnitude(self.player.vel);
        for pos in time_synchronized_points_on_line(
            start_pos,
            end_pos,
            start_time,
            end_time,
            1.0 / time_frequency_of_speed_particles,
        ) {
            self.place_particle_with_velocity(pos, rotated(particle_vel, -90.0));
            self.place_particle_with_velocity(pos, rotated(particle_vel, 90.0));
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

    fn get_occupancy_of_nearby_walls(&self, square: Point<i32>) -> AdjacentOccupancyMask {
        let mut output = [[false; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let rx = i as i32 - 1;
                let ry = j as i32 - 1;
                output[i][j] = matches!(self.try_get_block(square + p(rx, ry)), Some(Block::Wall));
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

    fn update_screen(
        &mut self,
        stdout: &mut MouseTerminal<termion::raw::RawTerminal<std::io::Stdout>>,
    ) {
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
        write!(stdout, "{}", termion::cursor::Goto(1, 1),).unwrap();
        stdout.flush().unwrap();
        self.output_on_screen = self.output_buffer.clone();
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
        self.set_block(snap_to_grid(self.player.pos), Block::Air);
        self.player.alive = false;
    }

    fn player_is_grabbing_wall(&self) -> bool {
        self.player_is_pressing_against_wall_horizontally()
            && self.player.vel.y() <= 0.0
            && !self.player_is_standing_on_block()
    }

    fn player_is_running_up_wall(&self) -> bool {
        self.player_is_pressing_against_wall_horizontally() && self.player.vel.y() > 0.0
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

    fn apply_player_acceleration_from_gravity(&mut self, dt_in_ticks: f32) {
        self.player
            .vel
            .add_assign(p(0.0, -self.player.acceleration_from_gravity * dt_in_ticks));
    }

    fn apply_player_kinematics(&mut self, dt_in_ticks: f32) {
        //dbg!(self.player.vel);
        let start_point = self.player.pos;
        let mut planned_displacement: Point<f32> =
            world_space_to_grid_space(self.player.vel, VERTICAL_STRETCH_FACTOR) * dt_in_ticks;
        let mut fraction_of_movement_remaining = 1.0;
        let mut current_start_point = start_point;
        let mut current_target = start_point + planned_displacement;
        let mut collision_occurred = false;
        loop {
            if let Some(collision) = self.unit_squarecast(current_start_point, current_target) {
                //dbg!(&current_start_point, &current_target, &collision);

                collision_occurred = true;

                let step_taken_to_this_collision = collision.collider_pos - current_start_point;
                let fraction_through_remaining_movement_just_moved =
                    magnitude(step_taken_to_this_collision) / magnitude(planned_displacement);
                fraction_of_movement_remaining -=
                    fraction_of_movement_remaining * fraction_through_remaining_movement_just_moved;

                self.player.last_collision = Some(PlayerBlockCollision {
                    time_in_ticks: self.time_in_ticks()
                        - dt_in_ticks * fraction_of_movement_remaining,
                    normal: collision.normal,
                    collider_velocity: self.player.vel,
                    collider_pos: collision.collider_pos,
                    collided_block_square: collision.collided_block_square,
                });

                (current_start_point, planned_displacement, self.player.vel) = self
                    .deflect_off_collision_plane(
                        collision,
                        current_start_point,
                        planned_displacement,
                        self.player.vel,
                    );

                current_target = current_start_point + planned_displacement;
            } else {
                // should exit loop after this else
                current_start_point = current_target;
                break;
            }
        }

        if !self.in_world(snap_to_grid(current_start_point)) {
            // Player went out of bounds and died
            self.kill_player();
            return;
        }

        let step_taken = current_start_point - self.player.pos;
        self.save_recent_player_pose(self.player.pos);
        self.player.pos = current_start_point;

        if collision_occurred {
            self.player.moved_normal_to_collision_since_collision = false;
        } else if self.player.last_collision.is_some()
            && step_taken.dot(floatify(
                self.player.last_collision.as_ref().unwrap().normal,
            )) != 0.0
        {
            self.player.moved_normal_to_collision_since_collision = true;
        }

        // moved vertically => instant empty charge
        if step_taken.y() != 0.0 {
            self.player.remaining_coyote_time = 0.0;
        }
        if self.player_is_standing_on_block() {
            self.player.remaining_coyote_time = self.player.max_coyote_time;
        } else if self.player.remaining_coyote_time > 0.0 {
            if self.player.remaining_coyote_time > dt_in_ticks {
                self.player.remaining_coyote_time -= dt_in_ticks;
            } else {
                self.player.remaining_coyote_time = 0.0;
            }
        }
    }

    fn deflect_off_collision_plane(
        &mut self,
        collision: SquarecastCollision,
        move_start: fpoint,
        relative_target: fpoint,
        vel: fpoint,
    ) -> (fpoint, fpoint, fpoint) {
        let step_taken_to_this_collision = collision.collider_pos - move_start;

        let mut new_relative_target = relative_target - step_taken_to_this_collision;
        let mut new_vel = vel;
        if collision.normal.x() != 0 {
            new_vel.set_x(0.0);
            new_relative_target = project(relative_target, right_f());
        } else if collision.normal.y() != 0 {
            new_vel.set_y(0.0);
            new_relative_target = project(relative_target, up_f());
        } else {
            panic!("collision has zero normal");
        }
        let new_move_start = collision.collider_pos;
        (new_move_start, new_relative_target, new_vel)
    }

    fn save_recent_player_pose(&mut self, pos: Point<f32>) {
        self.player.recent_poses.push_front(pos);
        while self.player.recent_poses.len() > NUM_SAVED_PLAYER_POSES as usize {
            self.player.recent_poses.pop_back();
        }
    }

    fn apply_forces_to_player(&mut self, dt_in_ticks: f32) {
        if self.player_is_grabbing_wall() && self.player.vel.y() <= 0.0 {
            self.apply_player_wall_friction(dt_in_ticks);
        } else if !self.player_is_supported() {
            self.apply_player_air_friction(dt_in_ticks);
            self.apply_player_air_traction(dt_in_ticks);
            self.apply_player_acceleration_from_gravity(dt_in_ticks);
        } else {
            self.apply_player_floor_friction(dt_in_ticks);
            self.apply_player_floor_traction(dt_in_ticks);
        }
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
    fn unit_squarecast(
        &self,
        start_pos: Point<f32>,
        end_pos: Point<f32>,
    ) -> Option<SquarecastCollision> {
        self.squarecast(start_pos, end_pos, 1.0)
    }

    fn linecast(&self, start_pos: Point<f32>, end_pos: Point<f32>) -> Option<SquarecastCollision> {
        self.squarecast(start_pos, end_pos, 0.0)
    }

    fn squarecast(
        &self,
        start_pos: Point<f32>,
        end_pos: Point<f32>,
        moving_square_side_length: f32,
    ) -> Option<SquarecastCollision> {
        let ideal_step = end_pos - start_pos;

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
            let mut collisions = Vec::<SquarecastCollision>::new();
            for overlapping_square in &overlapping_squares {
                if self.in_world(*overlapping_square)
                    && self.get_block(*overlapping_square) == Block::Wall
                {
                    let adjacent_occupancy =
                        self.get_occupancy_of_nearby_walls(*overlapping_square);
                    if let Some(collision) = single_block_squarecast_with_filled_cracks(
                        start_pos,
                        end_pos,
                        *overlapping_square,
                        moving_square_side_length,
                        adjacent_occupancy,
                    ) {
                        collisions.push(collision);
                    }
                }
            }
            if !collisions.is_empty() {
                collisions.sort_by(|a, b| {
                    let a_dist = a.collider_pos.euclidean_distance(&start_pos);
                    let b_dist = b.collider_pos.euclidean_distance(&start_pos);
                    a_dist.partial_cmp(&b_dist).unwrap()
                });
                let closest_collision_to_start = collisions[0];

                // might have missed one
                let normal_square = closest_collision_to_start.collided_block_square
                    + closest_collision_to_start.normal;
                if !point_inside_square(start_pos, normal_square)
                    && self.in_world(normal_square)
                    && self.get_block(normal_square) == Block::Wall
                {
                    let adjacent_occupancy = self.get_occupancy_of_nearby_walls(normal_square);
                    if let Some(collision) = single_block_squarecast_with_filled_cracks(
                        start_pos,
                        end_pos,
                        normal_square,
                        moving_square_side_length,
                        adjacent_occupancy,
                    ) {
                        //dbg!( start_pos, end_pos, normal_square, moving_square_side_length, adjacent_occupancy, &collision );
                        return Some(collision);
                    } else {
                        panic!("No collision with wall block normal to collision");
                    }
                }
                return Some(closest_collision_to_start);
            }
        }
        return None;
    }

    fn player_is_supported(&self) -> bool {
        return self.player_is_standing_on_block() || self.player.remaining_coyote_time > 0.0;
    }

    fn player_is_standing_on_block(&self) -> bool {
        self.player_exactly_touching_wall_in_direction(p(0, -1))
    }

    fn get_block_relative_to_player(&self, rel_pos: Point<i32>) -> Option<Block> {
        let target_pos = snap_to_grid(self.player.pos) + rel_pos;
        if self.player.alive && self.in_world(target_pos) {
            return Some(self.get_block(target_pos));
        }
        return None;
    }

    fn player_exactly_touching_wall_in_direction(&self, direction: Point<i32>) -> bool {
        for rel_x in -1..=1 {
            for rel_y in -1..=1 {
                let rel_square = p(rel_x, rel_y);
                if rel_square != p(0, 0) && direction.dot(rel_square) > 0 {
                    let target_square = snap_to_grid(self.player.pos) + rel_square;
                    if let Some(block) = self.get_block_relative_to_player(rel_square) {
                        if block == Block::Wall
                            && floating_square_orthogonally_touching_fixed_square(
                                self.player.pos,
                                target_square,
                            )
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn in_world(&self, pos: Point<i32>) -> bool {
        pos.x() >= 0
            && pos.x() < self.terminal_size.0 as i32
            && pos.y() >= 0
            && pos.y() < self.terminal_size.1 as i32
    }
}
fn init_world(width: u16, height: u16) -> Game {
    //let mut game = Game::new(width, height);
    let mut game = Game::new(10, 40);
    game.set_player_jump_delta_v(1.0);
    game.player.acceleration_from_gravity = 0.05;
    game.player.acceleration_from_floor_traction = 0.6;
    game.set_player_max_run_speed(0.7);

    game.place_boundary_wall();

    //let bottom_left = (
    //(game.terminal_size.0 * 2 / 5) as i32,
    //(game.terminal_size.1 / 4) as i32,
    //);
    //game.place_line_of_blocks(
    //bottom_left,
    //((4 * game.terminal_size.0 / 5) as i32, bottom_left.1),
    //Block::Wall,
    //);
    //game.place_line_of_blocks(
    //bottom_left,
    //(bottom_left.0, 3 * (game.terminal_size.1 / 4) as i32),
    //Block::Wall,
    //);
    game.place_player(
        game.terminal_size.0 as f32 / 2.0,
        game.terminal_size.1 as f32 / 2.0,
    );
    game.player.vel.set_x(0.1);
    game
}

fn main() {
    let stdin = stdin();
    let (width, height) = termion::terminal_size().unwrap();
    let mut game = init_world(width, height);
    // time saver
    //let mut game = Game::new(20, 40);
    let mut stdout = MouseTerminal::from(stdout().into_raw_mode().unwrap());

    write!(
        stdout,
        "{}{}q to exit.  c to clear.  Mouse to draw.  Begin!",
        termion::clear::All,
        termion::cursor::Goto(1, 1)
    )
    .unwrap();
    stdout.flush().unwrap();

    let (tx, rx) = channel();

    // Separate thread for reading input
    thread::spawn(move || {
        for c in stdin.events() {
            let evt = c.unwrap();
            tx.send(evt).unwrap();
        }
    });

    while game.running {
        let start_time = Instant::now();
        while let Ok(evt) = rx.try_recv() {
            game.handle_event(evt);
        }
        game.tick_physics();
        game.update_output_buffer();
        game.update_screen(&mut stdout);
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

    fn set_up_game() -> Game {
        Game::new(30, 30)
    }

    fn set_up_tall_drop() -> Game {
        let mut game = Game::new(20, 40);
        game.player.acceleration_from_gravity = 0.05;
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

    fn set_up_player_supported_by_coyote_frames() -> Game {
        let mut game = set_up_just_player();
        game.player.remaining_coyote_time = game.player.max_coyote_time;
        return game;
    }

    fn set_up_player_in_zero_g() -> Game {
        let mut game = set_up_just_player();
        game.player.acceleration_from_gravity = 0.0;
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
        game.player.desired_direction = p(1, 0);
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
        game.place_line_of_blocks((10, platform_y), (20, platform_y), Block::Wall);
        return game;
    }

    fn set_up_player_running_full_speed_to_right_on_platform() -> Game {
        let mut game = set_up_player_on_platform();
        game.player.vel = right_f() * game.player.max_run_speed;
        game.player.desired_direction = p(1, 0);
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

    fn set_up_player_flying_fast_through_space_in_direction(dir: Point<f32>) -> Game {
        let mut game = set_up_just_player();
        be_in_space(&mut game);
        let vel = direction(dir) * game.player.color_change_speed_threshold * 1.1;
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

    fn set_up_30_particles_about_to_move_one_square_right() -> Game {
        let mut game = set_up_game();
        let start_pos = p(0.49, 0.0);
        let start_vel = p(0.1, 0.0);
        for _ in 0..30 {
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

    fn be_frictionless(game: &mut Game) {
        game.player.acceleration_from_floor_traction = 0.0;
        game.player.deceleration_from_ground_friction = 0.0;
    }
    fn be_in_vacuum(game: &mut Game) {
        game.player.acceleration_from_air_traction = 0.0;
        game.player.deceleration_from_air_friction = 0.0;
    }
    fn be_in_zero_g(game: &mut Game) {
        game.player.acceleration_from_gravity = 0.0
    }
    fn be_in_space(game: &mut Game) {
        be_in_zero_g(game);
        be_in_vacuum(game);
    }
    fn be_in_frictionless_space(game: &mut Game) {
        be_frictionless(game);
        be_in_space(game);
    }

    #[test]
    #[timeout(100)]
    fn test_placement_and_gravity() {
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
        game.player.acceleration_from_gravity = 1.0;
        game.player.remaining_coyote_time = 0.0;
        game.tick_physics();
        assert!(game.player.alive == false);
    }

    #[test]
    #[timeout(100)]
    fn test_single_block_squarecast_no_move() {
        let point = p(0.0, 0.0);
        let p_wall = p(5, 5);

        assert!(single_block_unit_squarecast(point, point, p_wall) == None);
    }

    #[test]
    #[timeout(100)]
    fn test_squarecast_no_move() {
        let game = Game::new(30, 30);
        let point = p(0.0, 0.0);

        assert!(game.unit_squarecast(point, point) == None);
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

        assert!(result != None);
        assert!(points_nearly_equal(
            result.unwrap().collider_pos,
            floatify(p_wall) + p(-1.0, 0.0)
        ));
        assert!(result.unwrap().normal == p(-1, 0));
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

        assert!(result != None);
        assert!(points_nearly_equal(
            result.unwrap().collider_pos,
            floatify(p_wall) + p(0.0, -1.0)
        ));
        assert!(result.unwrap().normal == p(0, -1));
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

        assert!(result != None);
        assert!(points_nearly_equal(
            result.unwrap().collider_pos,
            floatify(p_wall) + p(0.0, 1.0)
        ));
        assert!(result.unwrap().normal == p(0, 1));
    }

    #[test]
    #[timeout(100)]
    fn test_unit_squarecast_to_upper_right() {
        let mut game = Game::new(30, 30);
        game.place_line_of_blocks((10, 10), (20, 10), Block::Wall);

        let squarecast_result = game.unit_squarecast(p(15.0, 9.0), p(17.0, 11.0)).unwrap();
        assert!(points_nearly_equal(
            squarecast_result.collider_pos,
            p(15.0, 9.0)
        ));
        assert!(squarecast_result.normal == p(0, -1));
        assert!(squarecast_result.collided_block_square.y() == 10);
    }

    #[test]
    #[timeout(100)]
    fn test_unit_squarecast_to_right() {
        let mut game = Game::new(30, 30);
        let wall_square = p(10, 10);
        game.place_wall_block(wall_square);

        let squarecast_result = game
            .unit_squarecast(
                floatify(wall_square + p(-5, 0)),
                floatify(wall_square + p(5, 0)),
            )
            .unwrap();
        assert!(points_nearly_equal(
            squarecast_result.collider_pos,
            floatify(wall_square) - p(1.0, 0.0)
        ));
        assert!(squarecast_result.normal == p(-1, 0));
        assert!(squarecast_result.collided_block_square.y() == wall_square.y());
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
        assert!(squarecast_result.is_none());
    }

    #[test]
    #[timeout(100)]
    fn test_unit_squarecast_to_far_upper_right() {
        let mut game = Game::new(30, 30);
        game.place_line_of_blocks((10, 10), (20, 10), Block::Wall);
        let squarecast_result = game.unit_squarecast(p(15.0, 9.0), p(17.0, 110.0)).unwrap();
        assert!(points_nearly_equal(
            squarecast_result.collider_pos,
            p(15.0, 9.0)
        ));
        assert!(squarecast_result.normal == p(0, -1));
        assert!(squarecast_result.collided_block_square.y() == 10);

        assert!(game.unit_squarecast(p(1.0, 9.0), p(-17.0, 9.0)) == None);
        assert!(game.unit_squarecast(p(15.0, 9.0), p(17.0, -11.0)) == None);
    }

    #[test]
    #[timeout(100)]
    fn test_squarecast_skips_player() {
        let game = set_up_player_on_platform();

        assert!(game.unit_squarecast(p(15.0, 11.0), p(15.0, 13.0)) == None);
    }

    #[test]
    #[timeout(100)]
    fn test_in_world_check() {
        let game = Game::new(30, 30);
        assert!(game.in_world(p(0, 0)));
        assert!(game.in_world(p(29, 29)));
        assert!(!game.in_world(p(30, 30)));
        assert!(!game.in_world(p(10, -1)));
        assert!(!game.in_world(p(-1, 10)));
    }

    #[test]
    #[timeout(100)]
    fn test_move_player() {
        let mut game = set_up_player_on_platform();
        game.player.max_run_speed = 1.0;
        game.player.desired_direction.set_x(1);

        game.tick_physics();

        assert!(game.player.pos == p(16.0, 11.0));

        game.place_player(15.0, 11.0);
        assert!(game.player.desired_direction.x() == 0);
        game.player.desired_direction.set_x(-1);

        game.tick_physics();
        game.tick_physics();

        assert!(game.player.pos == p(13.0, 11.0));
    }

    #[test]
    #[timeout(100)]
    fn test_stop_on_collision() {
        let mut game = set_up_player_on_platform();
        game.place_block(round(game.player.pos) + p(1, 0), Block::Wall);
        game.player.max_run_speed = 1.0;
        game.player.desired_direction.set_x(1);

        game.tick_physics();

        assert!(game.player.pos == p(15.0, 11.0));
        assert!(game.player.vel.x() == 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_snap_to_object_on_collision() {
        let mut game = set_up_player_on_platform();
        game.place_block(round(game.player.pos) + p(2, 0), Block::Wall);
        game.player.pos.add_assign(p(0.999, 0.0));
        game.player.desired_direction.set_x(1);
        game.player.vel.set_x(5.0);

        game.tick_physics();

        assert!(game.player.pos == p(16.0, 11.0));
        assert!(game.grid[17][11] == Block::Wall);
    }

    #[test]
    #[timeout(100)]
    fn test_move_player_slowly() {
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
        let mut game = set_up_player_on_platform();
        game.player.max_run_speed = 2.0;
        game.player.desired_direction.set_x(1);

        game.tick_physics();
        game.tick_physics();
        assert!(game.player.pos.x() > 17.0);
        game.tick_physics();
        assert!(game.player.pos.x() > 19.0);
    }

    #[test]
    #[timeout(100)]
    fn test_fast_player_collision_between_frames() {
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
        game.player.acceleration_from_gravity = 1.0;
        game.player.remaining_coyote_time = 0.0;

        game.tick_physics();

        assert!(game.player.pos.y() < 11.0);
    }

    #[test]
    #[timeout(100)]
    fn test_land_after_jump() {
        let mut game = set_up_player_on_platform();
        let start_pos = game.player.pos;

        game.player_jump();
        for _ in 0..50 {
            game.tick_physics();
        }
        assert!(nearly_equal(game.player.pos.y(), start_pos.y()));
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
        let mut game = Game::new(30, 30);
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
        game.player.acceleration_from_gravity = 0.0;
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
    #[timeout(100)]
    fn test_coyote_frames() {
        let mut game = set_up_just_player();
        let start_pos = game.player.pos;
        game.player.acceleration_from_gravity = 1.0;
        game.player.max_coyote_time = 1.0;
        game.player.remaining_coyote_time = 1.0;
        game.tick_physics();
        assert!(game.player.pos.y() == start_pos.y());
        game.tick_physics();
        assert!(game.player.pos.y() < start_pos.y());
    }

    #[test]
    #[timeout(100)]
    fn test_coyote_frames_dont_assist_jump() {
        let mut game1 = set_up_player_on_platform();
        let mut game2 = set_up_player_on_platform();

        game1.player.remaining_coyote_time = 0.0;
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
    fn test_player_does_not_spawn_with_coyote_frames() {
        let mut game = set_up_just_player();
        let start_pos = game.player.pos;
        game.player.acceleration_from_gravity = 1.0;
        game.tick_physics();
        assert!(game.player.pos.y() != start_pos.y());
    }

    #[test]
    #[timeout(100)]
    fn test_wall_jump_while_running_up_wall() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player.vel.set_y(1.0);
        game.player_jump_if_possible();
        assert!(game.player.vel.x() > 0.0);
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

    #[test]
    #[timeout(100)]
    fn test_slow_down_to_exactly_max_speed_horizontally_midair() {
        let mut game = set_up_player_barely_fighting_air_friction_to_the_right_in_zero_g();
        game.tick_physics();
        assert!(game.player.vel.x() == game.player.air_friction_start_speed);
    }

    #[test]
    #[timeout(100)]
    fn test_slow_down_to_exactly_max_speed_vertically_midair() {
        let mut game = set_up_player_barely_fighting_air_friction_up_in_zero_g();
        game.tick_physics();
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

    #[test]
    #[timeout(100)]
    fn test_braille_left_behind_when_go_fast() {
        let mut game = set_up_player_on_platform_in_box();
        game.player.vel = p(game.player.color_change_speed_threshold * 5.0, 0.0);
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
        assert!(game.player.recent_poses.is_empty());
    }

    #[test]
    #[timeout(100)]
    fn test_player_recent_poses_are_saved() {
        let mut game = set_up_player_on_platform();
        let p0 = game.player.pos;
        let p1 = p(5.0, 2.0);
        let p2 = p(6.7, 3.4);
        game.tick_physics();
        game.player.pos = p1;
        game.tick_physics();
        game.player.pos = p2;
        game.tick_physics();

        assert!(game.player.recent_poses.len() == 3);
        assert!(game.player.recent_poses.get(0).unwrap() == &p2);
        assert!(game.player.recent_poses.get(1).unwrap() == &p1);
        assert!(game.player.recent_poses.get(2).unwrap() == &p0);
    }

    #[test]
    #[timeout(100)]
    fn test_dash_sets_velocity_rather_than_adds_to_it_if_set() {
        let mut game = set_up_just_player();
        game.player.dash_adds_to_vel = false;
        game.player.vel = p(-game.player.dash_vel * 4.0, 0.0);
        game.player.desired_direction = p(1, 0);
        game.player_dash();
        assert!(game.player.vel.x() > 0.0);
    }

    #[test]
    #[timeout(100)]
    fn test_dash_adds_velocity_rather_than_sets_it_if_set() {
        let mut game = set_up_just_player();
        game.player.dash_adds_to_vel = true;
        game.player.vel = p(-game.player.dash_vel * 4.0, 0.0);
        game.player.desired_direction = p(1, 0);
        game.player_dash();
        assert!(game.player.vel.x() == -game.player.dash_vel * 3.0);
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

    #[test]
    #[timeout(100)]
    fn update_color_threshold_with_jump_delta_v_update() {
        let mut game = set_up_game();
        let start_thresh = game.player.color_change_speed_threshold;
        game.set_player_jump_delta_v(game.player.jump_delta_v + 1.0);
        assert!(game.player.color_change_speed_threshold != start_thresh);
    }

    #[test]
    #[timeout(100)]
    fn update_color_threshold_with_speed_update() {
        let mut game = set_up_game();
        let start_thresh = game.player.color_change_speed_threshold;
        game.set_player_max_run_speed(game.player.max_run_speed + 1.0);
        assert!(game.player.color_change_speed_threshold != start_thresh);
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
        assert!(!game.player_is_officially_fast());
    }

    #[test]
    #[timeout(100)]
    fn test_no_high_speed_color_with_normal_wall_jump() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_jump_if_possible();
        assert!(game.get_player_color() != PLAYER_HIGH_SPEED_COLOR);
        assert!(!game.player_is_officially_fast());
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
        let mut game = set_up_player_in_zero_g();
        game.player.acceleration_from_floor_traction = 0.0;
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
        game.player.acceleration_from_gravity = 1.0;
        let start_vy = game.player.vel.y();
        game.bullet_time_factor = 0.5;
        game.toggle_bullet_time();
        game.tick_physics();
        let expected_end_vy =
            -game.player.acceleration_from_gravity * game.bullet_time_factor + start_vy;
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
        game.player.color_change_speed_threshold = 9.0;

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
        assert!(movement.x() == movement.y() * VERTICAL_STRETCH_FACTOR);
    }

    #[test]
    #[timeout(100)]
    fn test_particles__movement_compensates_for_non_square_grid() {
        let mut game = set_up_game();
        game.place_particle_with_velocity_and_lifetime(p(5.0, 5.0), p(1.0, 1.0), 500.0);

        let start_pos = game.particles[0].pos;

        game.tick_physics();
        let movement = game.particles[0].pos - start_pos;
        assert!(movement.x() == movement.y() * VERTICAL_STRETCH_FACTOR);
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
        particle.random_walk_speed = DEFAULT_PARTICLE_STEP_PER_TICK;
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
        game1.particles[0].random_walk_speed = DEFAULT_PARTICLE_STEP_PER_TICK;
        game2.place_particle(start_pos);
        game2.particles[0].random_walk_speed = DEFAULT_PARTICLE_STEP_PER_TICK;

        game1.tick_particles();
        game2.tick_particles();

        let end_pos_1 = game1.particles[0].pos;
        let end_pos_2 = game2.particles[0].pos;
        let distorted_diff1 = end_pos_1 - start_pos;
        let distorted_diff2 = end_pos_2 - start_pos;
        let diff1 = grid_space_to_world_space(
            distorted_diff1 / game1.bullet_time_factor.sqrt(),
            VERTICAL_STRETCH_FACTOR,
        );
        let diff2 = grid_space_to_world_space(distorted_diff2, VERTICAL_STRETCH_FACTOR);

        assert!(end_pos_1 != end_pos_2);

        dbg!(diff1, magnitude(diff1), diff2, magnitude(diff2));
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
        be_in_frictionless_space(&mut game);
        assert!(&game.player.last_collision.is_none());
        let collision_velocity = p(0.0, -5.0);
        game.player.vel = collision_velocity;
        game.tick_physics();
        assert!(nearly_equal(
            game.time_since_last_player_collision().unwrap(),
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
            game.time_since_last_player_collision().unwrap(),
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
        assert!(game.time_since_last_player_collision() == Some(1.0));
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
            game.time_since_last_player_collision().unwrap(),
            ticks_before_bullet_time
        ));
        game.toggle_bullet_time();
        game.tick_physics();
        assert!(nearly_equal(
            game.time_since_last_player_collision().unwrap(),
            game.bullet_time_factor + ticks_before_bullet_time
        ));
        game.tick_physics();
        assert!(nearly_equal(
            game.time_since_last_player_collision().unwrap(),
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
            game.time_since_last_player_collision().unwrap(),
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
        dbg!(&chars_in_compression_start, &chars_in_compression_end);

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
            game.time_since_last_player_collision().unwrap(),
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
            game.time_since_last_player_collision().unwrap(),
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
            game.time_since_last_player_collision().unwrap(),
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
            game.time_since_last_player_collision().unwrap(),
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
        let mut game = set_up_player_touching_wall_on_right();
        game.player.vel = p(1.0, 0.0);
        game.player.desired_direction = p(1, 0);
        game.tick_physics();
        assert!(nearly_equal(
            game.time_since_last_player_collision().unwrap(),
            1.0
        ));
        assert!(game.player.moved_normal_to_collision_since_collision == false);
        game.player.vel = p(-1.0, 0.0);
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
    #[timeout(100)]
    fn test_coyote_frames_respect_bullet_time() {
        let mut game = set_up_player_supported_by_coyote_frames();
        assert!(game.player.remaining_coyote_time == game.player.max_coyote_time);
        game.toggle_bullet_time();
        game.tick_physics();
        assert!(game.player.remaining_coyote_time + 1.0 > game.player.max_coyote_time);
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
        let mut game = set_up_player_on_platform();
        be_in_frictionless_space(&mut game);
        game.player.pos.add_assign(p(0.0, 1.0));
        game.player.vel = p(0.0, -4.0);

        game.tick_physics();

        assert!(nearly_equal(
            game.time_since_last_player_collision().unwrap(),
            0.5
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

        let game1_time_before_impact = 1.0 - game1.time_since_last_player_collision().unwrap();
        let game2_time_before_impact = 1.0 - game2.time_since_last_player_collision().unwrap();
        assert!(game1_time_before_impact == game2_time_before_impact * VERTICAL_STRETCH_FACTOR);
    }

    #[test]
    #[timeout(100)]
    fn test_player_supported_by_block_if_mostly_off_edge() {
        let mut game = set_up_player_on_block_more_overhanging_than_not_on_right();
        game.player.remaining_coyote_time = 0.0;
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

    #[ignore]
    // like a spring
    #[test]
    #[timeout(100)]
    fn test_jump_bonus_if_jump_when_coming_out_of_compression() {}

    #[ignore]
    // just as the spring giveth, so doth the spring taketh away(-eth)
    #[test]
    #[timeout(100)]
    fn test_jump_penalty_if_jump_when_entering_compression() {}

    #[test]
    #[timeout(100)]
    fn test_perpendicular_speed_lines_move_perpendicular() {
        let dir = direction(p(1.25, 3.38)); // arbitrary
        let mut game = set_up_player_flying_fast_through_space_in_direction(dir);
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
        let mut game = set_up_30_particles_about_to_move_one_square_right();
        let start_square = p(0, 0);
        let particle_square = p(1, 0);
        game.tick_physics();
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
        let mut game1 = set_up_player_flying_fast_through_space_in_direction(dir);
        let mut game2 = set_up_player_flying_fast_through_space_in_direction(dir);
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
        let collision = game.squarecast(start_pos, start_pos + dir * 500.0, 1.0);
        assert!(points_nearly_equal(
            collision.unwrap().collider_pos,
            p(4.0, start_pos.y())
        ));
        assert!(collision.unwrap().collided_block_square == p(5, 5));
        assert!(collision.unwrap().normal == p(-1, 0));
    }
    #[test]
    #[timeout(100)]
    fn test_squarecast__hit_some_blocks_with_a_point() {
        let game = set_up_four_wall_blocks_at_5_and_6();
        let start_pos = p(3.0, 5.3);
        let dir = right_f();
        let collision = game.squarecast(start_pos, start_pos + dir * 500.0, 0.0);
        assert!(points_nearly_equal(
            collision.unwrap().collider_pos,
            p(4.5, start_pos.y())
        ));
        assert!(collision.unwrap().collided_block_square == p(5, 5));
        assert!(collision.unwrap().normal == p(-1, 0));
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
    fn test_linecast_hit_between_blocks() {
        let game = set_up_four_wall_blocks_at_5_and_6();
        let start_pos = p(10.0, 10.0);
        let end_pos = p(6.4999, 5.4999);
        let collision = game.linecast(start_pos, end_pos);
        assert!(collision.is_some());
        assert!(nearly_equal(collision.unwrap().collider_pos.x(), 6.5));
        assert!(collision.unwrap().collided_block_square == p(6, 6));
        assert!(collision.unwrap().normal == p(1, 0));
    }
    #[test]
    #[timeout(100)]
    fn test_linecast__hit_near_corner_at_grid_bottom() {
        let plus_center = p(7, 0);
        let game = set_up_plus_sign_wall_blocks_at_square(plus_center);
        let start_pos = floatify(plus_center) + p(-1.0, 1.0);
        let end_pos = floatify(plus_center) + p(0.0, -0.001);
        let collision = game.linecast(start_pos, end_pos);
        assert!(collision.is_some());
        assert!(collision.unwrap().collided_block_square == plus_center + p(-1, 0));
        assert!(collision.unwrap().normal == p(0, 1));
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
        let collision = game.linecast(start_pos, end_pos);
        dbg!(&collision);
        assert!(collision.is_some());
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
}
