#![feature(is_sorted)]
mod glyph;
mod utility;

extern crate device_query;
extern crate geo;
extern crate line_drawing;
extern crate num;
extern crate std;
extern crate termion;
#[macro_use]
extern crate approx;

// use assert2::{assert, check};
use crate::num::traits::Pow;
use device_query::{DeviceQuery, DeviceState, Keycode};
use geo::algorithm::euclidean_distance::EuclideanDistance;
use geo::algorithm::line_intersection::{line_intersection, LineIntersection};
use geo::{point, CoordNum, Point};
use std::char;
use std::cmp::min;
use std::collections::{HashSet, VecDeque};
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
const DEFAULT_PLAYER_ACCELERATION_FROM_TRACTION: f32 = 1.0;
const DEFAULT_PLAYER_GROUND_FRICTION_DECELERATION: f32 =
    DEFAULT_PLAYER_ACCELERATION_FROM_TRACTION / 5.0;
const DEFAULT_PLAYER_SOFT_MAX_RUN_SPEED: f32 = 0.5;
const DEFAULT_PLAYER_DASH_SPEED: f32 = DEFAULT_PLAYER_SOFT_MAX_RUN_SPEED * 3.0;
const DEFAULT_PLAYER_AIR_FRICTION_DECELERATION: f32 = 0.0;
const DEFAULT_PLAYER_MIDAIR_SOFT_MAX_SPEED: f32 = 5.0;

const DEFAULT_BULLET_TIME_FACTOR: f32 = 0.1;
const DEFAULT_PARTICLE_LIFETIME_IN_SECONDS: f32 = 0.5;
const DEFAULT_PARTICLE_LIFETIME_IN_TICKS: f32 =
    DEFAULT_PARTICLE_LIFETIME_IN_SECONDS * MAX_FPS as f32;
const DEFAULT_PARTICLE_STEP_PER_TICK: f32 = 0.1;
const DEFAULT_MAX_COMPRESSION: f32 = 0.2;
const DEFAULT_TICKS_TO_MAX_COMPRESSION: f32 = 5.0;
const DEFAULT_TICKS_TO_END_COMPRESSION: f32 = 10.0;

// These have no positional information
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Block {
    Air,
    Wall,
    Brick,
}

impl Block {
    fn character(&self) -> char {
        match self {
            Block::Air => ' ',
            Block::Wall => '█',
            Block::Brick => '▪',
        }
    }
    fn color(&self) -> ColorName {
        match self {
            Block::Air => ColorName::Black,
            Block::Wall => ColorName::White,
            Block::Brick => ColorName::White,
        }
    }

    fn subject_to_block_gravity(&self) -> bool {
        match self {
            Block::Air | Block::Wall => false,
            _ => true,
        }
    }
    fn wall_grabbable(&self) -> bool {
        match self {
            Block::Air => false,
            _ => true,
        }
    }
}

#[derive(PartialEq, Debug)]
struct Particle {
    ticks_to_expiry: f32,
    pos: Point<f32>,
    //vel: Point<f32>,
}

impl Particle {
    fn glyph(&self) -> Glyph {
        Glyph::world_pos_to_colored_braille_glyph(self.pos, ColorName::Blue)
    }
}

struct CollisionWithBlock {
    time_in_ticks: f32,
    normal: Point<i32>,
    collider_velocity: Point<f32>,
    collider_pos: Point<f32>,
    collided_block_square: Point<i32>,
}

struct Player {
    alive: bool,
    pos: Point<f32>,
    recent_poses: VecDeque<Point<f32>>,
    max_run_speed_bpf: f32,
    max_midair_speed: f32,
    color_change_speed_threshold: f32,
    vel_bpf: Point<f32>,
    desired_direction: Point<i32>,
    jump_delta_v: f32,
    acceleration_from_gravity: f32,
    acceleration_from_traction: f32,
    deceleration_from_air_friction: f32,
    deceleration_from_ground_friction: f32,
    remaining_coyote_time: f32,
    max_coyote_time: f32,
    dash_vel: f32,
    last_collision: Option<CollisionWithBlock>,
    moved_normal_to_collision_since_collision: bool,
}

impl Player {
    fn new() -> Player {
        Player {
            alive: false,
            pos: p(0.0, 0.0),
            recent_poses: VecDeque::<Point<f32>>::new(),
            max_run_speed_bpf: DEFAULT_PLAYER_SOFT_MAX_RUN_SPEED,
            max_midair_speed: DEFAULT_PLAYER_MIDAIR_SOFT_MAX_SPEED,
            color_change_speed_threshold: magnitude(p(
                DEFAULT_PLAYER_SOFT_MAX_RUN_SPEED,
                DEFAULT_PLAYER_JUMP_DELTA_V,
            )),
            vel_bpf: Point::<f32>::new(0.0, 0.0),
            desired_direction: p(0, 0),
            jump_delta_v: DEFAULT_PLAYER_JUMP_DELTA_V,
            acceleration_from_gravity: DEFAULT_PLAYER_ACCELERATION_FROM_GRAVITY,
            acceleration_from_traction: DEFAULT_PLAYER_ACCELERATION_FROM_TRACTION,
            deceleration_from_air_friction: DEFAULT_PLAYER_AIR_FRICTION_DECELERATION,
            deceleration_from_ground_friction: DEFAULT_PLAYER_GROUND_FRICTION_DECELERATION,
            remaining_coyote_time: DEFAULT_PLAYER_MAX_COYOTE_TIME,
            max_coyote_time: DEFAULT_PLAYER_MAX_COYOTE_TIME,
            dash_vel: DEFAULT_PLAYER_DASH_SPEED,
            last_collision: None,
            moved_normal_to_collision_since_collision: false,
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
            bullet_time_factor: DEFAULT_BULLET_TIME_FACTOR,
            player: Player::new(),
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
        self.player.max_run_speed_bpf = speed;
        self.update_player_color_change_speed_thresh();
    }

    fn update_player_color_change_speed_thresh(&mut self) {
        self.player.color_change_speed_threshold =
            magnitude(p(self.player.max_run_speed_bpf, self.player.jump_delta_v));
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
    fn get_block(&self, pos: Point<i32>) -> Block {
        return self.grid[pos.x() as usize][pos.y() as usize];
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
        self.place_line_of_blocks((0, 0), (xmax, 0), Block::Wall);
        self.place_line_of_blocks((xmax, 0), (xmax, ymax), Block::Wall);
        self.place_line_of_blocks((xmax, ymax), (0, ymax), Block::Wall);
        self.place_line_of_blocks((0, ymax), (0, 0), Block::Wall);
    }

    fn place_block(&mut self, pos: Point<i32>, block: Block) {
        self.grid[pos.x() as usize][pos.y() as usize] = block;
    }

    fn place_line_of_particles(&mut self, pos0: Point<f32>, pos1: Point<f32>) {
        let braille_pos0 = Glyph::world_pos_to_braille_pos(pos0);
        let braille_pos1 = Glyph::world_pos_to_braille_pos(pos1);
        for (x, y) in line_drawing::Bresenham::new(
            snap_to_grid(braille_pos0).x_y(),
            snap_to_grid(braille_pos1).x_y(),
        ) {
            let particle_pos = Glyph::braille_pos_to_world_pos(p(x as f32, y as f32));
            self.place_particle(particle_pos);
        }
    }

    fn place_particle(&mut self, pos: Point<f32>) {
        self.place_particle_with_lifespan(pos, DEFAULT_PARTICLE_LIFETIME_IN_TICKS)
    }

    fn place_particle_with_lifespan(&mut self, pos: Point<f32>, lifespan_in_ticks: f32) {
        self.particles.push(Particle {
            pos,
            ticks_to_expiry: lifespan_in_ticks,
        })
    }

    fn get_particles_in_square(&self, square: Point<i32>) -> Vec<&Particle> {
        self.particles
            .iter()
            .filter(|particle| snap_to_grid(particle.pos) == square)
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
        self.player.vel_bpf = p(0.0, 0.0);
        self.player.desired_direction = p(0, 0);
        self.player.pos = p(x, y);
        self.player.alive = true;
        self.player.remaining_coyote_time = 0.0;
    }
    fn player_jump(&mut self) {
        if self.player_is_grabbing_wall() || self.player_is_running_up_wall() {
            let wall_direction = sign(self.player.desired_direction);
            self.player
                .vel_bpf
                .add_assign(floatify(-wall_direction) * self.player.max_run_speed_bpf);
            self.player.desired_direction = -wall_direction;
        }
        self.player.vel_bpf.set_y(self.player.jump_delta_v);
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
            self.player.vel_bpf = floatify(self.player.desired_direction) * self.player.dash_vel;
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
                self.make_speed_lines();
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
        self.particles.iter_mut().for_each(|particle| {
            particle.ticks_to_expiry -= dt_in_ticks;
            // sqrt because 2d random walk (maybe not quite right)
            particle.pos.add_assign(
                direction(random_direction()) * DEFAULT_PARTICLE_STEP_PER_TICK * dt_in_ticks.sqrt(),
            );
        });

        self.particles
            .retain(|particle| particle.ticks_to_expiry > 0.0);

        self.delete_particles_out_of_world();
    }

    fn delete_particles_out_of_world(&mut self) {
        let particles_are_in_world: Vec<bool> = self
            .particles
            .iter()
            .map(|particle| self.in_world(snap_to_grid(particle.pos)))
            .collect();

        particles_are_in_world
            .into_iter()
            .enumerate()
            .filter(|(i, is_in_world)| !*is_in_world)
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
        let actual_speed = magnitude(self.player.vel_bpf);
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
                if self.grid[x][y] != Block::Air {
                    self.output_buffer[x][y] = Glyph::from_char(self.grid[x][y].character());
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

    fn make_speed_lines(&mut self) {
        if let Some(&last_pos) = self.player.recent_poses.get(0) {
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
                self.place_line_of_particles(last_pos + offset, self.player.pos + offset);
            }
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

    fn get_buffered_glyph(&self, pos: Point<i32>) -> &Glyph {
        return &self.output_buffer[pos.x() as usize][pos.y() as usize];
    }
    fn set_buffered_glyph(&mut self, pos: Point<i32>, new_glyph: Glyph) {
        self.output_buffer[pos.x() as usize][pos.y() as usize] = new_glyph;
    }
    fn get_glyph_on_screen(&self, pos: Point<i32>) -> &Glyph {
        return &self.output_on_screen[pos.x() as usize][pos.y() as usize];
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
            && self.player.vel_bpf.y() <= 0.0
            && !self.player_is_standing_on_block()
    }

    fn player_is_running_up_wall(&self) -> bool {
        self.player_is_pressing_against_wall_horizontally() && self.player.vel_bpf.y() > 0.0
    }

    fn player_is_pressing_against_wall_horizontally(&self) -> bool {
        if self.player.desired_direction.x() == 0 {
            return false;
        }
        let horizontal_desired_direction = p(self.player.desired_direction.x(), 0);
        if let Some(block) = self.get_block_relative_to_player(horizontal_desired_direction) {
            return block.wall_grabbable();
        }
        return false;
    }

    fn apply_player_wall_friction(&mut self, dt_in_ticks: f32) {
        self.player.vel_bpf.set_y(decelerate_linearly_to_cap(
            self.player.vel_bpf.y(),
            0.0,
            self.player.acceleration_from_traction * dt_in_ticks,
        ));
    }

    fn apply_player_floor_traction(&mut self, dt_in_ticks: f32) {
        self.player.vel_bpf.set_x(accelerate_within_max_speed(
            self.player.vel_bpf.x(),
            self.player.desired_direction.x(),
            self.player.max_run_speed_bpf,
            self.player.acceleration_from_traction * dt_in_ticks,
        ));
    }

    fn apply_player_air_traction(&mut self, dt_in_ticks: f32) {
        self.player.vel_bpf.set_x(accelerate_within_max_speed(
            self.player.vel_bpf.x(),
            self.player.desired_direction.x(),
            self.player.max_run_speed_bpf,
            self.player.acceleration_from_traction * dt_in_ticks,
        ));
    }

    fn apply_player_acceleration_from_gravity(&mut self, dt_in_ticks: f32) {
        self.player
            .vel_bpf
            .add_assign(p(0.0, -self.player.acceleration_from_gravity * dt_in_ticks));
    }

    fn apply_player_kinematics(&mut self, dt_in_ticks: f32) {
        let start_point = self.player.pos;
        let mut remaining_step: Point<f32> =
            compensate_for_vertical_stretch(self.player.vel_bpf, VERTICAL_STRETCH_FACTOR)
                * dt_in_ticks;
        let mut current_start_point = start_point;
        let actual_endpoint: Point<f32>;
        let mut current_target = start_point + remaining_step;
        let mut collision_occurred = false;
        loop {
            if let Some(collision) = self.movecast(current_start_point, current_target) {
                collision_occurred = true;
                remaining_step.add_assign(-(collision.collider_pos - current_start_point));

                self.player.last_collision = Some(CollisionWithBlock {
                    time_in_ticks: self.time_in_ticks(),
                    normal: collision.normal,
                    collider_velocity: self.player.vel_bpf,
                    collider_pos: collision.collider_pos,
                    collided_block_square: collision.collided_block_square,
                });

                if collision.normal.x() != 0 {
                    self.player.vel_bpf.set_x(0.0);
                    remaining_step = project(remaining_step, p(0.0, 1.0));
                }
                if collision.normal.y() != 0 {
                    self.player.vel_bpf.set_y(0.0);
                    remaining_step = project(remaining_step, p(1.0, 0.0));
                }

                current_start_point = collision.collider_pos;
                current_target = current_start_point + remaining_step;
            } else {
                // should exit loop after this else
                actual_endpoint = current_target;
                break;
            }
        }

        if !self.in_world(snap_to_grid(actual_endpoint)) {
            // Player went out of bounds and died
            self.kill_player();
            return;
        }

        let step_taken = actual_endpoint - self.player.pos;
        self.save_recent_player_pose(self.player.pos);
        self.player.pos = actual_endpoint;

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

    fn save_recent_player_pose(&mut self, pos: Point<f32>) {
        self.player.recent_poses.push_front(pos);
        while self.player.recent_poses.len() > NUM_SAVED_PLAYER_POSES as usize {
            self.player.recent_poses.pop_back();
        }
    }

    fn apply_forces_to_player(&mut self, dt_in_ticks: f32) {
        if self.player_is_grabbing_wall() && self.player.vel_bpf.y() <= 0.0 {
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
        self.player.vel_bpf.set_x(decelerate_linearly_to_cap(
            self.player.vel_bpf.x(),
            self.player.max_midair_speed,
            self.player.deceleration_from_air_friction * dt_in_ticks,
        ));

        self.player.vel_bpf.set_y(decelerate_linearly_to_cap(
            self.player.vel_bpf.y(),
            self.player.max_midair_speed,
            self.player.deceleration_from_air_friction,
        ));
    }

    fn apply_player_floor_friction(&mut self, dt_in_ticks: f32) {
        self.player.vel_bpf.set_x(decelerate_linearly_to_cap(
            self.player.vel_bpf.x(),
            self.player.max_run_speed_bpf,
            self.player.deceleration_from_ground_friction * dt_in_ticks,
        ));
    }

    // Where the player can move to in a line
    // tries to draw a line in air
    // returns None if out of bounds
    // returns the start position if start is not Block::Air
    fn movecast(&self, start_pos: Point<f32>, end_pos: Point<f32>) -> Option<MovecastCollision> {
        let ideal_step = end_pos - start_pos;

        let collision_checks_per_block_travelled =
            self.num_positions_per_block_to_check_for_collisions;
        let num_points_to_check =
            (magnitude(ideal_step) * collision_checks_per_block_travelled).floor() as i32;
        let mut intermediate_player_positions_to_check = Vec::<Point<f32>>::new();
        for i in 0..num_points_to_check {
            intermediate_player_positions_to_check.push(
                start_pos
                    + direction(ideal_step) * (i as f32 / collision_checks_per_block_travelled),
            );
        }
        // needed for very small steps
        intermediate_player_positions_to_check.push(end_pos);
        for point_to_check in &intermediate_player_positions_to_check {
            let overlapping_squares =
                grid_squares_overlapped_by_floating_unit_square(*point_to_check);
            let mut collisions = Vec::<MovecastCollision>::new();
            for overlapping_square in &overlapping_squares {
                if self.in_world(*overlapping_square)
                    && self.get_block(*overlapping_square) == Block::Wall
                {
                    if let Some(collision) =
                        single_block_movecast(start_pos, *point_to_check, *overlapping_square)
                    {
                        collisions.push(collision);
                    } else {
                        panic!("Started inside a block");
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
                return Some(closest_collision_to_start);
            }
        }
        return None;
    }

    fn player_is_supported(&self) -> bool {
        return self.player_is_standing_on_block() || self.player.remaining_coyote_time > 0.0;
    }

    fn player_is_standing_on_block(&self) -> bool {
        !matches!(self.get_block_below_player(), None | Some(Block::Air))
            && offset_from_grid(self.player.pos).y() == 0.0
    }

    fn get_block_below_player(&self) -> Option<Block> {
        return self.get_block_relative_to_player(p(0, -1));
    }

    fn get_block_relative_to_player(&self, rel_pos: Point<i32>) -> Option<Block> {
        let target_pos = snap_to_grid(self.player.pos) + rel_pos;
        if self.player.alive && self.in_world(target_pos) {
            return Some(self.get_block(target_pos));
        }
        return None;
    }

    fn in_world(&self, pos: Point<i32>) -> bool {
        return pos.x() >= 0
            && pos.x() < self.terminal_size.0 as i32
            && pos.y() >= 0
            && pos.y() < self.terminal_size.1 as i32;
    }
    fn init_world(&mut self) {
        self.set_player_jump_delta_v(1.0);
        self.player.acceleration_from_gravity = 0.05;
        self.player.acceleration_from_traction = 0.6;
        self.set_player_max_run_speed(0.7);

        self.place_boundary_wall();

        let bottom_left = (
            (self.terminal_size.0 / 5) as i32,
            (self.terminal_size.1 / 4) as i32,
        );
        self.place_line_of_blocks(
            bottom_left,
            ((4 * self.terminal_size.0 / 5) as i32, bottom_left.1),
            Block::Wall,
        );
        self.place_line_of_blocks(
            bottom_left,
            (bottom_left.0, 3 * (self.terminal_size.1 / 4) as i32),
            Block::Wall,
        );
        self.place_player(
            self.terminal_size.0 as f32 / 2.0,
            self.terminal_size.1 as f32 / 2.0,
        );
    }
}

fn main() {
    let stdin = stdin();
    let (width, height) = termion::terminal_size().unwrap();
    let mut game = Game::new(width, height);
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

    // time saver
    game.init_world();

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

    fn set_up_player_in_zero_g_frictionless_vacuum() -> Game {
        let mut game = set_up_player_in_zero_g();
        game.player.acceleration_from_traction = 0.0;
        game.player.deceleration_from_ground_friction = 0.0;
        game.player.deceleration_from_air_friction = 0.0;
        return game;
    }

    fn set_up_player_on_platform() -> Game {
        let mut game = set_up_just_player();
        let platform_y = game.player.pos.y() as i32 - 1;
        game.place_line_of_blocks((10, platform_y), (20, platform_y), Block::Wall);
        return game;
    }

    fn set_up_player_on_platform_in_frictionless_vacuum() -> Game {
        let mut game = set_up_player_on_platform();
        game.player.acceleration_from_traction = 0.0;
        game.player.deceleration_from_air_friction = 0.0;
        return game;
    }

    fn set_up_player_under_platform() -> Game {
        let mut game = set_up_just_player();
        let platform_y = game.player.pos.y() as i32 + 1;
        game.place_line_of_blocks((10, platform_y), (20, platform_y), Block::Wall);
        return game;
    }

    fn set_up_player_left_of_wall() -> Game {
        let mut game = set_up_just_player();
        let wall_x = game.player.pos.x() as i32 + 1;
        game.place_line_of_blocks((wall_x, 0), (wall_x, 20), Block::Wall);
        return game;
    }

    fn set_up_player_in_corner_of_big_L() -> Game {
        let mut game = set_up_player_on_platform();
        game.place_line_of_blocks((14, 10), (14, 20), Block::Wall);
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
        game.place_line_of_blocks((14, 0), (14, 20), Block::Wall);
        game.player.desired_direction.set_x(-1);
        return game;
    }

    #[test]
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
    fn test_player_dies_when_falling_off_screen() {
        let mut game = Game::new(30, 30);
        game.place_player(15.0, 0.0);
        game.player.acceleration_from_gravity = 1.0;
        game.player.remaining_coyote_time = 0.0;
        game.tick_physics();
        assert!(game.player.alive == false);
    }

    #[test]
    fn test_single_block_movecast_no_move() {
        let point = p(0.0, 0.0);
        let p_wall = p(5, 5);

        assert!(single_block_movecast(point, point, p_wall) == None);
    }

    #[test]
    fn test_movecast_no_move() {
        let game = Game::new(30, 30);
        let point = p(0.0, 0.0);

        assert!(game.movecast(point, point) == None);
    }

    #[test]
    fn test_movecast_horizontal_hit() {
        let mut game = Game::new(30, 30);
        let p_wall = p(5, 0);
        game.place_block(p_wall, Block::Wall);

        let p1 = floatify(p_wall) + p(-2.0, 0.0);
        let p2 = floatify(p_wall) + p(2.0, 0.0);
        let result = game.movecast(p1, p2);

        assert!(result != None);
        assert!(result.unwrap().collider_pos == floatify(p_wall) + p(-1.0, 0.0));
        assert!(result.unwrap().normal == p(-1, 0));
    }

    #[test]
    fn test_movecast_vertical_hit() {
        let mut game = Game::new(30, 30);
        let p_wall = p(15, 10);
        game.place_block(p_wall, Block::Wall);

        let p1 = floatify(p_wall) + p(0.0, -1.1);
        let p2 = floatify(p_wall);
        let result = game.movecast(p1, p2);

        assert!(result != None);
        assert!(result.unwrap().collider_pos == floatify(p_wall) + p(0.0, -1.0));
        assert!(result.unwrap().normal == p(0, -1));
    }

    #[test]
    fn test_movecast_end_slightly_overlapping_a_block() {
        let mut game = Game::new(30, 30);
        let p_wall = p(15, 10);
        game.place_block(p_wall, Block::Wall);

        let p1 = floatify(p_wall) + p(0.0, 1.5);
        let p2 = floatify(p_wall) + p(0.0, 0.999999);
        let result = game.movecast(p1, p2);

        assert!(result != None);
        assert!(result.unwrap().collider_pos == floatify(p_wall) + p(0.0, 1.0));
        assert!(result.unwrap().normal == p(0, 1));
    }

    #[test]
    fn test_movecast() {
        let mut game = Game::new(30, 30);
        game.place_line_of_blocks((10, 10), (20, 10), Block::Wall);

        assert!(
            game.movecast(p(15.0, 9.0), p(17.0, 11.0))
                == Some(MovecastCollision {
                    collider_pos: p(15.0, 9.0),
                    normal: p(0, -1),
                    collided_block_square: p(15, 10),
                })
        );
        assert!(
            game.movecast(p(15.0, 9.0), p(17.0, 110.0))
                == Some(MovecastCollision {
                    collider_pos: p(15.0, 9.0),
                    normal: p(0, -1),
                    collided_block_square: p(15, 10),
                })
        );
        assert!(game.movecast(p(1.0, 9.0), p(-17.0, 9.0)) == None);
        assert!(game.movecast(p(15.0, 9.0), p(17.0, -11.0)) == None);
    }

    #[test]
    fn test_movecast_ignore_player() {
        let game = set_up_player_on_platform();

        assert!(game.movecast(p(15.0, 11.0), p(15.0, 13.0)) == None);
    }

    #[test]
    fn test_in_world_check() {
        let game = Game::new(30, 30);
        assert!(game.in_world(p(0, 0)));
        assert!(game.in_world(p(29, 29)));
        assert!(!game.in_world(p(30, 30)));
        assert!(!game.in_world(p(10, -1)));
        assert!(!game.in_world(p(-1, 10)));
    }

    #[test]
    fn test_move_player() {
        let mut game = set_up_player_on_platform();
        game.player.max_run_speed_bpf = 1.0;
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
    fn test_stop_on_collision() {
        let mut game = set_up_player_on_platform();
        game.place_block(round(game.player.pos) + p(1, 0), Block::Wall);
        game.player.max_run_speed_bpf = 1.0;
        game.player.desired_direction.set_x(1);

        game.tick_physics();

        assert!(game.player.pos == p(15.0, 11.0));
        assert!(game.player.vel_bpf.x() == 0.0);
    }

    #[test]
    fn test_snap_to_object_on_collision() {
        let mut game = set_up_player_on_platform();
        game.place_block(round(game.player.pos) + p(2, 0), Block::Wall);
        game.player.pos.add_assign(p(0.999, 0.0));
        game.player.desired_direction.set_x(1);
        game.player.vel_bpf.set_x(5.0);

        game.tick_physics();

        assert!(game.player.pos == p(16.0, 11.0));
        assert!(game.grid[17][11] == Block::Wall);
    }

    #[test]
    fn test_move_player_slowly() {
        let mut game = set_up_player_on_platform();
        game.player.max_run_speed_bpf = 0.49;
        game.player.acceleration_from_traction = 999.9; // rilly fast
        game.player.desired_direction.set_x(1);

        game.tick_physics();

        assert!(game.player.pos.x() > 15.0);
        assert!(game.player.pos.x() < 15.5);

        game.tick_physics();

        assert!(game.player.pos.x() > 15.5);
        assert!(game.player.pos.x() < 16.0);
    }

    #[test]
    fn test_move_player_quickly() {
        let mut game = set_up_player_on_platform();
        game.player.max_run_speed_bpf = 2.0;
        game.player.desired_direction.set_x(1);

        game.tick_physics();
        game.tick_physics();
        assert!(game.player.pos.x() > 17.0);
        game.tick_physics();
        assert!(game.player.pos.x() > 19.0);
    }

    #[test]
    fn test_fast_player_collision_between_frames() {
        let mut game = set_up_player_on_platform();
        // Player should not teleport through this block
        game.place_block(p(16, 11), Block::Wall);
        game.player.max_run_speed_bpf = 2.0;
        game.player.desired_direction.set_x(1);

        game.tick_physics();
        assert!(game.player.pos == p(15.0, 11.0));
        assert!(game.player.vel_bpf.x() == 0.0);
    }

    #[test]
    fn test_can_jump() {
        let mut game = set_up_player_on_platform();
        let start_pos = game.player.pos;

        game.player_jump();
        game.tick_physics();
        assert!(game.player.pos.y() > start_pos.y());
    }

    #[test]
    fn test_player_gravity() {
        let mut game = Game::new(30, 30);
        game.place_player(15.0, 11.0);
        game.player.acceleration_from_gravity = 1.0;
        game.player.remaining_coyote_time = 0.0;

        game.tick_physics();

        assert!(game.player.pos.y() < 11.0);
    }

    #[test]
    fn test_land_after_jump() {
        let mut game = set_up_player_on_platform();
        let start_pos = game.player.pos;

        game.player_jump();
        for _ in 0..50 {
            game.tick_physics();
        }
        assert!(game.player.pos.y() == start_pos.y());
    }

    #[test]
    fn test_slide_on_angled_collision() {
        let mut game = Game::new(30, 30);
        game.place_line_of_blocks((10, 10), (20, 10), Block::Wall);
        game.place_line_of_blocks((14, 10), (14, 20), Block::Wall);

        game.place_player(15.0, 11.0);
        game.player.vel_bpf = p(-2.0, 2.0);
        game.tick_physics();
        assert!(game.player.vel_bpf.x() == 0.0);
        assert!(game.player.vel_bpf.y() > 0.0);
    }

    #[test]
    fn test_decellerate_when_already_moving_faster_than_max_speed() {
        let mut game = set_up_player_on_platform();
        game.player.max_run_speed_bpf = 1.0;
        game.player.vel_bpf.set_x(5.0);
        game.player.desired_direction.set_x(1);
        game.tick_physics();
        assert!(game.player.vel_bpf.x() > game.player.max_run_speed_bpf);
        assert!(game.player.vel_bpf.x() < 5.0);
    }

    #[test]
    fn test_no_double_jump() {
        let mut game = set_up_player_on_platform();
        game.player_jump_if_possible();
        game.tick_physics();
        let vel_y_before_second_jump = game.player.vel_bpf.y();
        game.player_jump_if_possible();
        game.tick_physics();
        assert!(game.player.vel_bpf.y() < vel_y_before_second_jump);
    }

    #[test]
    fn test_respawn_button() {
        let mut game = Game::new(30, 30);
        game.handle_event(Event::Key(Key::Char('r')));
        assert!(game.player.alive);
        assert!(game.player.pos == p(15.0, 15.0));
    }

    #[test]
    fn test_wall_grab() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.tick_physics();
        assert!(game.player.vel_bpf.y() == 0.0);
    }

    #[test]
    fn test_simple_wall_jump() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_jump_if_possible();
        game.tick_physics();
        assert!(game.player.vel_bpf.y() > 0.0);
        assert!(game.player.vel_bpf.x() > 0.0);
    }

    #[test]
    fn test_no_running_up_walls_immediately_after_spawn() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player.vel_bpf = p(-1.0, 1.0);
        let start_vel_y = game.player.vel_bpf.y();
        game.tick_physics();
        assert!(game.player.vel_bpf.y() < start_vel_y);
    }

    #[test]
    fn test_dont_grab_wall_while_moving_up() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player.vel_bpf.set_y(1.0);
        assert!(!game.player_is_grabbing_wall());
    }

    #[test]
    fn test_no_friction_when_sliding_up_wall() {
        let mut game = set_up_player_on_platform();
        let wall_x = game.player.pos.x() as i32 - 2;
        game.place_line_of_blocks((wall_x, 0), (wall_x, 20), Block::Wall);
        game.player.desired_direction.set_x(-1);
        game.player.pos.add_assign(p(0.0, 2.0));
        game.player.pos.set_x(wall_x as f32 + 1.0);
        game.player.vel_bpf = p(-3.0, 3.0);
        game.player.acceleration_from_gravity = 0.0;
        game.player.deceleration_from_air_friction = 0.0;

        let start_y_vel = game.player.vel_bpf.y();
        let start_y_pos = game.player.pos.y();
        game.tick_physics();
        let end_y_vel = game.player.vel_bpf.y();
        let end_y_pos = game.player.pos.y();

        assert!(start_y_vel == end_y_vel);
        assert!(start_y_pos != end_y_pos);
    }

    #[test]
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

        assert!(game1.player.vel_bpf.y() == game2.player.vel_bpf.y());
    }

    #[test]
    fn test_player_does_not_spawn_with_coyote_frames() {
        let mut game = set_up_just_player();
        let start_pos = game.player.pos;
        game.player.acceleration_from_gravity = 1.0;
        game.tick_physics();
        assert!(game.player.pos.y() != start_pos.y());
    }

    #[test]
    fn test_wall_jump_while_running_up_wall() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player.vel_bpf.set_y(1.0);
        game.player_jump_if_possible();
        assert!(game.player.vel_bpf.x() > 0.0);
    }

    #[test]
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
    // The timing of when to slide to a stop should give the player precision positioning
    fn test_dont_snap_to_grid_when_sliding_to_a_halt() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(0.1, 0.0));
        game.player.vel_bpf = p(0.1, 0.0);
        game.player.acceleration_from_traction = 0.1;
        game.player.desired_direction = p(0, 0);

        game.tick_physics();
        assert!(game.player.vel_bpf.x() == 0.0);
        assert!(offset_from_grid(game.player.pos).x() != 0.0);
    }

    #[ignore]
    #[test]
    // The general case of this is pressing jump any time before landing causing an instant jump when possible
    fn test_allow_early_jump() {}

    #[ignore]
    #[test]
    // The general case of this is allowing a single jump anytime while falling after walking (not jumping!) off a platform
    fn test_allow_late_jump() {}

    #[ignore]
    #[test]
    // This should allow high skill to lead to really fast wall climbing (like in N+)
    fn test_wall_jump_adds_velocity_instead_of_sets_it() {}

    #[ignore]
    #[test]
    // Once we have vertical subsquare positioning up and running, a slow slide down will look cool.
    fn test_slowly_slide_down_when_grabbing_wall() {}

    #[test]
    fn test_dash_no_direction_does_nothing() {
        let mut game = set_up_player_on_platform();
        let start_vel = game.player.vel_bpf;
        game.player_dash();
        assert!(game.player.vel_bpf == start_vel);
    }

    #[test]
    fn test_dash_right_on_ground() {
        let mut game = set_up_player_on_platform();
        let start_vel = game.player.vel_bpf;
        game.player.desired_direction = p(1, 0);
        game.player_dash();
        assert!(
            game.player.vel_bpf == floatify(game.player.desired_direction) * game.player.dash_vel
        );
    }

    #[test]
    fn test_slow_down_to_exactly_max_speed_horizontally_midair() {
        let mut game = set_up_just_player();
        game.player.desired_direction = p(1, 0);
        game.player.max_midair_speed = 10.0;
        game.player.deceleration_from_air_friction = 1.0;
        game.player.acceleration_from_traction = 0.0;
        let vx = game.player.max_midair_speed + game.player.deceleration_from_air_friction * 0.9;
        game.player.vel_bpf = p(vx, 0.0);
        game.tick_physics();
        assert!(game.player.vel_bpf.x() == game.player.max_midair_speed);
    }

    #[test]
    fn test_slow_down_to_exactly_max_speed_vertically_midair() {
        let mut game = set_up_player_in_zero_g();
        game.player.desired_direction = p(0, 1);
        game.player.max_midair_speed = 10.0;
        game.player.deceleration_from_air_friction = 1.0;
        let vy = game.player.max_midair_speed + game.player.deceleration_from_air_friction * 0.9;
        game.player.vel_bpf = p(0.0, vy);
        game.tick_physics();
        assert!(game.player.vel_bpf.y() == game.player.max_midair_speed);
    }

    #[test]
    fn test_air_friction_has_no_slowdown_overshoot() {
        let mut game = set_up_player_in_zero_g();
        game.player.desired_direction = p(1, 0);
        game.player.max_midair_speed = 10.0;
        game.player.deceleration_from_air_friction = 1.0;
        game.player.acceleration_from_traction = 0.0;
        let start_vx =
            game.player.max_midair_speed + game.player.deceleration_from_air_friction * 0.9;
        game.player.vel_bpf = p(start_vx, 0.0);
        game.tick_physics();
        let vx_after_slowdown = game.player.vel_bpf.x();
        game.tick_physics();
        let end_vx = game.player.vel_bpf.x();
        assert!(vx_after_slowdown < start_vx);
        assert!(end_vx == vx_after_slowdown);
    }

    #[test]
    fn test_do_not_grab_wall_while_standing_on_ground() {
        let mut game = set_up_player_on_platform();
        let wall_x = game.player.pos.x() as i32 - 1;
        game.place_line_of_blocks((wall_x, 0), (wall_x, 20), Block::Wall);
        game.player.desired_direction = p(-1, 0);

        assert!(game.player_is_standing_on_block() == true);
        assert!(game.player_is_grabbing_wall() == false);
    }

    #[test]
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

    #[test]
    fn test_different_color_when_go_fast() {
        let mut game = set_up_player_on_platform();
        let stopped_color = game.get_player_glyphs()[1][1].clone().unwrap().fg_color;
        game.player.vel_bpf = p(game.player.max_run_speed_bpf + 5.0, 0.0);
        let fast_color = game.get_player_glyphs()[1][1].clone().unwrap().fg_color;

        assert!(stopped_color != fast_color);
    }

    #[test]
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
    fn test_braille_speed_lines_when_go_fast() {
        let mut game = set_up_player_on_platform_in_box();
        game.player.vel_bpf = p(game.player.color_change_speed_threshold * 5.0, 0.0);
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
    fn test_player_recent_poses_starts_empty() {
        let mut game = set_up_player_on_platform();
        assert!(game.player.recent_poses.is_empty());
    }

    #[test]
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
    fn test_dash_sets_velocity_rather_than_adds_to_it() {
        let mut game = set_up_just_player();
        game.player.vel_bpf = p(-game.player.dash_vel * 4.0, 0.0);
        game.player.desired_direction = p(1, 0);
        game.player_dash();
        assert!(game.player.vel_bpf.x() > 0.0);
    }

    #[test]
    fn test_prevent_clip_into_wall_when_colliding_with_internal_block_edge() {
        // data comes from observed bug reproduction
        let start_pos = p(35.599968, 16.5);
        let prev_pos = p(36.899967, 17.25);

        let mut game = Game::new(50, 50);
        game.player.deceleration_from_air_friction = 0.0;
        game.place_player(start_pos.x(), start_pos.y());
        game.player.vel_bpf = start_pos - prev_pos;
        game.player.desired_direction = p(-1, 0);

        game.place_block(p(34, 17), Block::Wall);
        game.place_block(p(34, 16), Block::Wall);
        game.place_block(p(34, 15), Block::Wall);

        game.num_positions_per_block_to_check_for_collisions = 0.00001;
        game.tick_physics();

        assert!(game.player.pos.x() == 35.0);
    }

    #[test]
    fn update_color_threshold_with_jump_delta_v_update() {
        let mut game = set_up_game();
        let start_thresh = game.player.color_change_speed_threshold;
        game.set_player_jump_delta_v(game.player.jump_delta_v + 1.0);
        assert!(game.player.color_change_speed_threshold != start_thresh);
    }

    #[test]
    fn update_color_threshold_with_speed_update() {
        let mut game = set_up_game();
        let start_thresh = game.player.color_change_speed_threshold;
        game.set_player_max_run_speed(game.player.max_run_speed_bpf + 1.0);
        assert!(game.player.color_change_speed_threshold != start_thresh);
    }

    #[test]
    fn test_no_high_speed_color_with_normal_move_and_jump() {
        let mut game = set_up_player_on_platform();
        game.player.vel_bpf = p(game.player.max_run_speed_bpf, 0.0);
        game.player.desired_direction = p(1, 0);
        game.player_jump();
        assert!(game.get_player_color() == PLAYER_COLOR);
        assert!(game.get_player_color() != PLAYER_HIGH_SPEED_COLOR);
        assert!(!game.player_is_officially_fast());
    }

    #[test]
    fn test_no_high_speed_color_with_normal_wall_jump() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player_jump_if_possible();
        assert!(game.get_player_color() != PLAYER_HIGH_SPEED_COLOR);
        assert!(!game.player_is_officially_fast());
    }

    #[ignore]
    // ignore because midair max speed may be higher than running max speed
    #[test]
    fn test_be_fastest_at_very_start_of_jump() {
        let mut game = set_up_player_on_platform();
        game.player.vel_bpf = p(game.player.max_run_speed_bpf, 0.0);
        game.player.desired_direction = p(1, 0);
        game.player_jump();
        let vel_at_start_of_jump = game.player.vel_bpf;
        game.tick_physics();
        game.tick_physics();
        game.tick_physics();

        assert!(magnitude(game.player.vel_bpf) <= magnitude(vel_at_start_of_jump));
    }

    #[test]
    fn test_toggle_bullet_time() {
        let mut game = set_up_player_on_platform();
        assert!(!game.is_bullet_time);
        game.handle_event(Event::Key(Key::Char('g')));
        assert!(game.is_bullet_time);
        game.handle_event(Event::Key(Key::Char('g')));
        assert!(!game.is_bullet_time);
    }

    #[test]
    fn test_bullet_time_slows_down_motion() {
        let mut game = set_up_player_in_zero_g();
        game.player.acceleration_from_traction = 0.0;
        game.player.vel_bpf = p(1.0, 0.0);
        let start_x = game.player.pos.x();
        game.bullet_time_factor = 0.5;
        game.toggle_bullet_time();
        game.tick_physics();
        let expected_end_x = game.player.vel_bpf.x() * game.bullet_time_factor + start_x;
        assert!(game.player.pos.x() == expected_end_x);
    }

    #[test]
    fn test_bullet_time_slows_down_acceleration_from_gravity() {
        let mut game = set_up_just_player();
        game.player.acceleration_from_gravity = 1.0;
        let start_vy = game.player.vel_bpf.y();
        game.bullet_time_factor = 0.5;
        game.toggle_bullet_time();
        game.tick_physics();
        let expected_end_vy =
            -game.player.acceleration_from_gravity * game.bullet_time_factor + start_vy;
        assert!(game.player.vel_bpf.y() == expected_end_vy);
    }

    #[ignore]
    // The real puzzler here is the accelerations that go exactly to a speed and no further.  They make it hard to factor things out.
    #[test]
    fn test_bullet_time_slows_down_traction_acceleration() {
        let mut game1 = set_up_player_starting_to_move_right_on_platform();
        let mut game2 = set_up_player_starting_to_move_right_on_platform();
        game1.toggle_bullet_time();

        game1.tick_physics();
        game2.tick_physics();

        assert!(game1.player.vel_bpf.x() < game2.player.vel_bpf.x());
    }

    #[test]
    fn test_no_passing_max_speed_horizontally_in_midair() {
        let mut game = set_up_just_player();
        game.player.vel_bpf = p(game.player.max_run_speed_bpf, 0.0);
        game.player.desired_direction = p(1, 0);
        game.tick_physics();
        assert!(game.player.vel_bpf.x().abs() <= game.player.max_midair_speed);
    }

    #[test]
    fn test_can_turn_around_midair_around_after_a_wall_jump() {
        let mut game = set_up_player_hanging_on_wall_on_left();
        game.player.deceleration_from_air_friction = 0.0;
        let start_vx = game.player.vel_bpf.x();
        game.player_jump_if_possible();
        let vx_at_start_of_jump = game.player.vel_bpf.x();
        game.tick_physics();
        let vx_one_frame_into_jump = game.player.vel_bpf.x();
        game.player.desired_direction = p(-1, 0);
        game.tick_physics();
        let end_vx = game.player.vel_bpf.x();

        assert!(start_vx == 0.0);
        assert!(vx_at_start_of_jump > 0.0);
        //assert!(vx_one_frame_into_jump == vx_at_start_of_jump);
        assert!(end_vx < vx_one_frame_into_jump);
    }

    #[test]
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
    fn do_not_hang_onto_ceiling() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(0.0, -2.0));
        game.player.desired_direction = p(0, 1);
        let start_y_pos = game.player.pos.y();
        game.tick_physics();
        assert!(game.player.pos.y() < start_y_pos);
    }

    #[test]
    fn test_movement_compensates_for_non_square_grid() {
        let mut game = set_up_player_in_zero_g_frictionless_vacuum();
        let start_pos = game.player.pos;

        game.player.vel_bpf = p(1.0, 1.0);
        game.tick_physics();
        let movement = game.player.pos - start_pos;
        assert!(movement.x() == movement.y() * VERTICAL_STRETCH_FACTOR);
    }

    #[ignore]
    #[test]
    fn test_fps_display() {}

    #[test]
    fn test_no_jump_if_not_touching_floor() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(0.0, 0.1));
        assert!(!game.player_can_jump());
    }

    #[test]
    fn test_player_not_standing_on_block_if_slightly_above_block() {
        let mut game = set_up_player_on_platform();
        game.player.pos.add_assign(p(0.0, 0.1));
        assert!(!game.player_is_standing_on_block());
    }

    #[test]
    fn test_place_particle() {
        let mut game = set_up_game();
        let pos = p(5.0, 5.0);
        game.place_particle(pos);
        assert!(game.particles.len() == 1);
        assert!(game.particles[0].pos == pos);
    }
    #[test]
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
    fn test_speed_lines_are_particles() {
        let mut game = set_up_just_player();
        game.player.desired_direction = p(1, 0);
        let start_pos = game.player.pos;
        game.player_dash();
        game.tick_physics();
        assert!(!game.particles.is_empty());
    }

    #[test]
    fn test_get_particles_in_square() {
        let mut game = set_up_game();
        game.place_particle(p(0.0, 0.0));
        game.place_particle(p(5.0, 5.0));
        game.place_particle(p(5.1, 5.0));
        game.place_particle(p(6.5, 5.0));
        game.place_particle(p(6.5, 5.0));
        game.place_particle(p(6.5, 5.0));
        game.place_particle(p(6.5, 5.0));

        assert!(game.get_particles_in_square(p(10, 10)).len() == 0);
        assert!(game.get_particles_in_square(p(0, 0)).len() == 1);
        assert!(game.get_particles_in_square(p(5, 5)).len() == 2);
        assert!(game.get_particles_in_square(p(6, 5)).len() == 0);
        assert!(game.get_particles_in_square(p(7, 5)).len() == 4);
        assert!(game.get_particles_in_square(p(-7, -5)).len() == 0);
    }
    #[test]
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
    fn test_particle_lines_can_visually_fill_square() {
        let mut game = set_up_game();
        game.place_line_of_particles(p(5.1, 6.0), p(5.1, 4.0));
        game.place_line_of_particles(p(4.9, 6.0), p(4.9, 4.0));
        game.update_output_buffer();
        assert!(game.get_particles_in_square(p(5, 5)).len() == 8);
        assert!(game.count_braille_dots_in_square(p(5, 5)) == 8);
    }
    #[test]
    fn test_particles_can_expire() {
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
    fn test_particles_move_around() {
        let point = p(5.0, 5.0);
        let mut game = set_up_game();
        game.place_particle(point);
        let start_pos = game.particles[0].pos;
        game.tick_particles();
        let end_pos = game.particles[0].pos;
        assert!(start_pos != end_pos);
    }

    #[test]
    fn test_particles_die_when_out_of_map() {
        let mut game = set_up_game();
        let point = p(50.0, 50.0);
        game.place_particle(point);
        game.delete_particles_out_of_world();
        assert!(game.particles.is_empty());
    }

    #[test]
    fn test_particle_movement_slowed_by_bullet_time() {
        let mut game1 = set_up_game();
        let mut game2 = set_up_game();
        game1.toggle_bullet_time();
        let start_pos = p(5.0, 5.0);
        game1.place_particle(start_pos);
        game2.place_particle(start_pos);

        game1.tick_particles();
        game2.tick_particles();

        let end_pos_1 = game1.particles[0].pos;
        let end_pos_2 = game2.particles[0].pos;
        let diff1 = end_pos_1 - start_pos;
        let diff2 = end_pos_2 - start_pos;

        assert!(
            (magnitude(diff1) / game1.bullet_time_factor.sqrt() - magnitude(diff2)).abs() < 0.0001
        );
    }

    #[test]
    fn test_track_last_collision() {
        let mut game = set_up_player_in_corner_of_big_L();
        game.player.acceleration_from_gravity = 0.0;
        game.player.deceleration_from_ground_friction = 0.0;
        game.player.pos.add_assign(p(0.01, 0.01));
        assert!(&game.player.last_collision.is_none());
        let collision_velocity = p(0.0, -5.0);
        game.player.vel_bpf = collision_velocity;
        game.tick_physics();
        assert!(game.time_since_last_player_collision() == Some(0.0));
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
        assert!(game.time_since_last_player_collision() == Some(1.0));
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
        game.player.vel_bpf = collision_velocity;
        game.tick_physics();
        assert!(game.time_since_last_player_collision() == Some(0.0));
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
    fn test_last_collision_tracking_accounts_for_bullet_time() {
        let mut game = set_up_player_on_platform();
        game.player.vel_bpf = p(0.0, -1.0);
        game.tick_physics();
        assert_relative_eq!(game.time_since_last_player_collision().unwrap(), 0.0);
        game.toggle_bullet_time();
        game.tick_physics();
        assert_relative_eq!(
            game.time_since_last_player_collision().unwrap(),
            game.bullet_time_factor
        );
        game.tick_physics();
        assert_relative_eq!(
            game.time_since_last_player_collision().unwrap(),
            game.bullet_time_factor * 2.0
        );
    }

    #[test]
    fn test_player_compresses_like_a_spring_when_colliding_at_high_speed() {
        let mut game = set_up_player_on_platform();
        game.player.vel_bpf = p(0.0, -10.0);
        game.tick_physics();
        game.tick_physics();
        assert!(game.get_player_compression_fraction() < 1.0);
        assert!(game.get_player_compression_fraction() > 0.0);
    }

    #[test]
    fn test_player_compression_overrides_visuals() {
        let mut game = set_up_just_player();
        game.player.last_collision = Some(CollisionWithBlock {
            time_in_ticks: game.time_in_ticks() - DEFAULT_TICKS_TO_MAX_COMPRESSION / 2.0,
            normal: p(0, 1),
            collider_velocity: p(0.0, -3.0),
            collider_pos: p(5.0, 5.0),
            collided_block_square: p(5, 4),
        });
        let player_glyphs = game.get_player_glyphs();
        assert!(
            EIGHTH_BLOCKS_FROM_BOTTOM[1..8].contains(&game.get_compressed_player_glyph().character)
        );
    }

    #[test]
    fn test_compress_horizontally_on_wall_collision() {
        let mut game = set_up_just_player();
        game.player.last_collision = Some(CollisionWithBlock {
            time_in_ticks: game.time_in_ticks() - DEFAULT_TICKS_TO_MAX_COMPRESSION / 2.0,
            normal: p(1, 0),
            collider_velocity: p(0.0, -3.0),
            collider_pos: p(5.0, 5.0),
            collided_block_square: p(5, 4),
        }); // parts of this don't add up, but they shouldn't need to
        let player_glyphs = game.get_player_glyphs();
        assert!(
            EIGHTH_BLOCKS_FROM_LEFT[1..8].contains(&game.get_compressed_player_glyph().character)
        );
    }

    #[test]
    fn test_player_compression_characters_for_floor_collision_appear_in_correct_sequence() {
        let mut game = set_up_player_on_platform();
        game.player.vel_bpf = p(0.0, -10.0);
        game.tick_physics();
        assert!(game.player.last_collision.is_some());
        assert!(game.time_since_last_player_collision().unwrap() == 0.0);

        let mut chars_in_compression_start = Vec::<char>::new();
        for t in 0..DEFAULT_TICKS_TO_MAX_COMPRESSION as i32 {
            chars_in_compression_start.push(game.get_compressed_player_glyph().character);
            game.tick_physics();
        }
        chars_in_compression_start.push(game.get_compressed_player_glyph().character);

        let mut chars_in_compression_end = Vec::<char>::new();
        for t in DEFAULT_TICKS_TO_MAX_COMPRESSION as i32..=DEFAULT_TICKS_TO_END_COMPRESSION as i32 {
            chars_in_compression_end.push(game.get_compressed_player_glyph().character);
            game.tick_physics();
        }
        dbg!(&chars_in_compression_start, &chars_in_compression_end);

        assert!(*chars_in_compression_start.first().unwrap() == EIGHTH_BLOCKS_FROM_BOTTOM[8]);
        assert!(*chars_in_compression_end.last().unwrap() == EIGHTH_BLOCKS_FROM_BOTTOM[8]);
        assert!(
            chars_in_compression_start.last().unwrap() == chars_in_compression_end.first().unwrap()
        );

        // reverse because of unicode char ordering
        chars_in_compression_start.reverse();
        assert!(chars_in_compression_start.is_sorted());
        assert!(chars_in_compression_end.is_sorted());
    }
    #[ignore]
    // maybe not a good idea?
    #[test]
    fn test_collision_compression_direction_based_on_collision_velocity_rather_than_collision_normal_on_ground(
    ) {
        let mut game = set_up_just_player();
        //game.player.normal_of_last_collision = Some(p(0, 1));
        //game.player.ticks_from_last_collision = Some(DEFAULT_TICKS_TO_MAX_COMPRESSION / 2.0);
        //game.player.velocity_of_last_collision = Some(p(10.0, -5.0));
        let chars_for_horizontal_compression = &EIGHTH_BLOCKS_FROM_LEFT[1..8];
        let char_of_compressed_player = game.get_compressed_player_glyph().character;
        assert!(chars_for_horizontal_compression.contains(&char_of_compressed_player));
    }

    #[ignore]
    // maybe not a good idea?
    #[test]
    fn test_collision_compression_direction_based_on_collision_velocity_rather_than_collision_normal_on_wall(
    ) {
        let mut game = set_up_just_player();
        //game.player.normal_of_last_collision = Some(p(-1, 0));
        //game.player.ticks_from_last_collision = Some(DEFAULT_TICKS_TO_MAX_COMPRESSION / 2.0);
        //game.player.velocity_of_last_collision = Some(p(4.0, -5.0));
        let chars_for_vertical_compression = &EIGHTH_BLOCKS_FROM_BOTTOM[1..8];
        let char_of_compressed_player = game.get_compressed_player_glyph().character;
        assert!(chars_for_vertical_compression.contains(&char_of_compressed_player));
    }

    #[test]
    fn test_leaving_floor_cancels_vertical_compression() {
        // Assignment
        let mut game = set_up_player_on_platform();
        //game.player.pos.add_assign(p(0.0, 0.1));
        game.player.vel_bpf = p(0.0, -10.0);
        game.tick_physics();
        game.tick_physics();
        assert!(game.time_since_last_player_collision() == Some(1.0));
        assert!(game.get_player_compression_fraction() < 1.0);

        // Action
        game.player_jump_if_possible();
        game.tick_physics();

        // Assertion
        assert!(game.get_player_compression_fraction() == 1.0);
    }
    #[test]
    fn test_leaving_ceiling_cancels_vertical_compression() {
        // Assignment
        let mut game = set_up_player_under_platform();
        //game.player.pos.add_assign(p(0.0, 0.1));
        game.player.vel_bpf = p(0.0, 10.0);
        game.tick_physics();
        assert!(game.time_since_last_player_collision() == Some(0.0));

        // Action
        game.tick_physics();

        // Assertion
        assert!(game.get_player_compression_fraction() == 1.0);
    }
    #[test]
    fn test_leaving_wall_cancels_horizontal_compression() {
        // Assignment
        let mut game = set_up_player_left_of_wall();
        game.player.acceleration_from_gravity = 0.0;
        game.player.acceleration_from_traction = 0.0;
        //game.player.pos.add_assign(p(0.0, 0.1));
        game.player.vel_bpf = p(10.0, 0.0);
        game.tick_physics();
        game.tick_physics();
        assert!(game.time_since_last_player_collision() == Some(1.0));
        assert!(game.get_player_compression_fraction() < 1.0);

        // Action
        game.player.vel_bpf = p(-0.1, 0.0);
        game.tick_physics();

        // Assertion
        assert!(game.get_player_compression_fraction() == 1.0);
    }

    #[test]
    fn test_particle_visuals_do_not_cover_blocks() {
        let mut game = set_up_game();
        let square = p(5, 5);

        game.place_particle(floatify(square));
        game.place_block(square, Block::Wall);
        game.update_output_buffer();

        let drawn_char = game.get_buffered_glyph(square).character;

        assert!(drawn_char == Block::Wall.character());
    }

    #[ignore]
    // like a spring
    #[test]
    fn test_jump_bonus_if_jump_when_coming_out_of_compression() {}

    #[ignore]
    // just as the spring giveth, so doth the spring taketh away(-eth)
    #[test]
    fn test_jump_penalty_if_jump_when_entering_compression() {}

    #[test]
    fn test_player_remembers_movement_normal_to_last_collision__vertical_collision() {
        let mut game = set_up_player_on_platform();
        game.player.vel_bpf = p(0.0, -1.0);
        game.tick_physics();
        assert!(game.time_since_last_player_collision() == Some(0.0));
        assert!(game.player.moved_normal_to_collision_since_collision == false);
        game.player_jump_if_possible();
        game.tick_physics();
        assert!(game.player.moved_normal_to_collision_since_collision == true);
    }

    #[test]
    fn test_player_remembers_movement_normal_to_last_collision__horizontal_collision() {
        let mut game = set_up_player_left_of_wall();
        game.player.vel_bpf = p(1.0, 0.0);
        game.tick_physics();
        assert!(game.time_since_last_player_collision() == Some(0.0));
        assert!(game.player.moved_normal_to_collision_since_collision == false);
        game.player.vel_bpf = p(-1.0, 0.0);
        game.tick_physics();
        assert!(game.player.moved_normal_to_collision_since_collision == true);
    }

    #[test]
    fn test_player_forgets_normal_movement_to_collision_upon_new_collision() {
        let mut game = set_up_player_on_platform_in_frictionless_vacuum();
        game.player.moved_normal_to_collision_since_collision = true;
        game.player.vel_bpf = p(0.0, -1.0);
        game.tick_physics();
        assert!(game.player.moved_normal_to_collision_since_collision == false);
    }

    #[test]
    fn test_coyote_frames_respect_bullet_time() {
        let mut game = set_up_player_supported_by_coyote_frames();
        assert!(game.player.remaining_coyote_time == game.player.max_coyote_time);
        game.toggle_bullet_time();
        game.tick_physics();
        assert!(game.player.remaining_coyote_time + 1.0 > game.player.max_coyote_time);
    }
}
