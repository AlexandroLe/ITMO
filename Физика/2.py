import pygame
import math
import sys

# Настройки
WIDTH, HEIGHT = 800, 800
FPS = 60
ROWS = 10          # количество рядов в треугольнике
SOLDIER_RADIUS = 5
MOVE_SPEED = 200
REPULSION_FORCE = 150
COLLISION_DIST = SOLDIER_RADIUS * 2.2

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

class Soldier:
    def __init__(self, x, y, factor):
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.factor = factor

    def update(self, dt):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)
        if dist > 0.1:
            step = min(self.factor * MOVE_SPEED * dt, dist)
            self.x += dx / dist * step
            self.y += dy / dist * step

    def apply_repulsion(self, others, dt):
        for other in others:
            if other is self: continue
            dx = self.x - other.x
            dy = self.y - other.y
            d = math.hypot(dx, dy)
            if 0 < d < COLLISION_DIST:
                overlap = COLLISION_DIST - d
                push = REPULSION_FORCE * overlap * dt / d
                self.x += dx * push
                self.y += dy * push

    def draw(self):
        radius = int(self.factor * SOLDIER_RADIUS) + 2
        pygame.draw.circle(screen, (0, 150, 200), (int(self.x), int(self.y)), radius)

# Генерируем треугольную формацию
soldiers = []
center_x, center_y = WIDTH // 2, HEIGHT // 2
spacing = SOLDIER_RADIUS * 4
count = 0
total = (ROWS * (ROWS + 1)) // 2  # общее число солдат в треугольнике

for i in range(ROWS):
    cols = i + 1
    for j in range(cols):
        offset_x = (j - (cols - 1) / 2) * spacing
        offset_y = i * spacing
        x = center_x + offset_x
        y = center_y + offset_y - (ROWS * spacing) / 2
        factor = 1 - (count / total)
        soldiers.append(Soldier(x, y, factor))
        count += 1

# Функция движения формации

def move_formation(dx, dy):
    for s in soldiers:
        s.target_x += dx * s.factor
        s.target_y += dy * s.factor

running = True
while running:
    dt = clock.tick(FPS) / 1000
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    dx = dy = 0
    if keys[pygame.K_LEFT]: dx = -100 * dt
    if keys[pygame.K_RIGHT]: dx = 100 * dt
    if keys[pygame.K_UP]: dy = -100 * dt
    if keys[pygame.K_DOWN]: dy = 100 * dt
    if dx or dy:
        move_formation(dx, dy)

    for s in soldiers:
        s.apply_repulsion(soldiers, dt)
    for s in soldiers:
        s.update(dt)

    screen.fill((30, 30, 30))
    for s in soldiers:
        s.draw()
    pygame.display.flip()

pygame.quit()
sys.exit()
