import pygame
import numpy as np

class Visuals:
    def __init__(self, quad):
        pygame.init()

        self.width = 500
        self.height = 500

        self.screen = pygame.display.set_mode(
            (self.width, self.height)
        )
        pygame.display.set_caption("Tuad")
        self.clock = pygame.time.Clock()

        self.origin = np.array([
            self.width / 2,
            self.height / 2
        ])

        self.quad = quad
        self.l = quad.l
        self.scale = 50
        self.running = True

        self.dt = 0.01
        self.accumulator = 0.0
        self.simTime = 0.0

    def world_to_screen(self, x, z):
        screen_x = self.origin[0] + x * self.scale
        screen_y = self.origin[1] - z * self.scale
        return np.array([int(screen_x), int(screen_y)])

    def draw(self):
        self.screen.fill((0, 0, 0))

        pygame.draw.line(self.screen, (80, 80, 80), (0, self.height // 2), (self.width, self.height // 2), 1)
        pygame.draw.line(self.screen, (80, 80, 80), (self.width // 2, 0), (self.width // 2, self.height), 1)

        x, z, vx, vz, theta, omega, TL, TR = self.quad.state
        center = self.world_to_screen(x, z)

        direction = np.array([np.cos(theta), np.sin(theta)]) # drone arm direction

        # motor positions
        left_world = np.array([x, z]) - self.l * direction
        right_world = np.array([x, z]) + self.l * direction

        left = self.world_to_screen(left_world[0], left_world[1])
        right = self.world_to_screen(right_world[0], right_world[1])

        # Tuad
        pygame.draw.line(self.screen, (0, 255, 0), left, right, 2)

        # motors
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            left + (0, 1.0),
            2
        )

        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            right + (0, 1.0),
            2
        )
        pygame.display.flip()

    def update(self, action):

        frameTime = self.clock.tick(60) / 1000.0
        self.accumulator += frameTime

        while self.accumulator >= self.dt:
            self.quad.step(action, self.dt)
            self.accumulator -= self.dt
            self.simTime += self.dt
        self.draw()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def close(self):
        pygame.quit()