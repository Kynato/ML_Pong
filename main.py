import pygame
import neat
import random
import math
import os

# STATICS
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
BAR_SIZE = 200
BALL_SIZE = 30
FPS = 120 
GAME_WINDOW = pygame.display.set_mode( (WINDOW_WIDTH, WINDOW_HEIGHT ))

pygame.font.init()

class Bar:
    HEIGHT = BAR_SIZE
    WIDTH = 30
    VEL = 5
    def __init__(self, x, spacing):
        self.x = x
        self.top = WINDOW_HEIGHT/2 - self.HEIGHT/2
        self.bottom = self.top + self.HEIGHT
        self.spacing = spacing
        self.color = pygame.Color(200, 200, 200)

    def draw(self, win):
        bar = pygame.Rect(self.x + self.spacing, self.top, self.WIDTH, self.HEIGHT)
        pygame.draw.rect(win, self.color, bar)

    def move(self, dir):
        if dir > 0.25:
            if self.top - self.VEL >= 0:
                # move up
                self.top -= self.VEL
                self.bottom = self.top + self.HEIGHT
        elif dir < -0.25:
            if self.bottom + self.VEL <= WINDOW_HEIGHT:
                # move down
                self.top += self.VEL
                self.bottom = self.top + self.HEIGHT
    
    def center(self):
        l = self.bottom - self.HEIGHT/2
        return l

class Ball:
    SIZE = BALL_SIZE
    VEL = 10
    BOUNCE_COOLDOWN = 10
    def __init__(self):
        self.x = WINDOW_WIDTH/2 - self.SIZE/2
        self.y = WINDOW_HEIGHT/2 - self.SIZE/2
        self.color = (255, 255, 255)
        self.angle = random.randrange(-44, 44)
        self.bounce_cooldown = self.BOUNCE_COOLDOWN

    def move(self):
        self.bounce_cooldown -= 1
        rad = math.radians(self.angle)
        new_x = self.x + (self.VEL*math.cos(rad))
        new_y = self.y + (self.VEL*math.sin(rad))

        self.x = new_x
        self.y = new_y
        self.ball = None

    def draw(self, win):
        ball = pygame.Rect(self.x, self.y, self.SIZE, self.SIZE)
        pygame.draw.rect(win, self.color, ball)

    def check_win(self):
        if self.x <= 0:
            return 'right'
        elif self.x >= WINDOW_WIDTH:
            return 'left'

        return None

    def random_angle(self):
        self.angle += random.randrange(-20, 20)

    def collision(self, left, right): 
        if self.y <= 0 or self.y + self.SIZE >= WINDOW_HEIGHT:
            self.angle = 360 - self.angle
        if self.bounce_cooldown > 0:
            return
        if self.x <= left.spacing + left.WIDTH:
            if self.y > left.top and self.y < left.bottom:
                # Lewy odbija
                self.angle = 180 - self.angle
                self.bounce_cooldown = self.BOUNCE_COOLDOWN
                return True

        elif self.x + self.SIZE >= WINDOW_WIDTH - right.WIDTH + right.spacing:
            if self.y > right.top and self.y < right.bottom:
                # Prawy odbija
                self.angle = 180 - self.angle
                self.bounce_cooldown = self.BOUNCE_COOLDOWN
                return True

    

    def going_left(self):
        if abs(self.angle) > 90 and (self.angle) < 270:
            return True
        else:
            return False
        
class Background:
    def __init__(self, color=None):
        if not color:
            self.color = (0, 0, 0)
        else:
            self.color = color
        self.fitness = 0
        self.STAT_FONT = pygame.font.SysFont("arial", 50)
        self.gen = 0
        self.species = 0

    def draw(self, win):
        # Draw background
        bg = pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(win, self.color, bg)

        # Score
        text_fitness = self.STAT_FONT.render("fitness: " + str(self.fitness), True, pygame.Color('white'))
        win.blit(text_fitness, (WINDOW_WIDTH - 10 - text_fitness.get_width(), 10))

        # Generation
        text_gen = self.STAT_FONT.render("generation: " + str(self.gen), True, pygame.Color("white"))
        win.blit(text_gen, (WINDOW_WIDTH/2 - text_gen.get_width()/2, 10))

        # Species left
        text_spec = self.STAT_FONT.render("species: " + str(self.species), True, pygame.Color("white"))
        win.blit(text_spec, (WINDOW_WIDTH/2 - text_spec.get_width()/2, WINDOW_HEIGHT - 10 - text_spec.get_height()))

# Draws the graphics
def game_refresh(win, bg, lb, rb, ball):

    bg.draw(win)

    lb.draw(win)
    rb.draw(win)

    ball.draw(win)

    pygame.display.update()

bg = Background() 

# Globals for UI
gen = 0
spec = 0

# Main learning loop
def eval_genomes(genomes, config):
    global gen
    global spec
    gen += 1
    spec = len(genomes)

    nets = []
    ge = []
    l_bars = []
    r_bars = []
    balls = []
    
    
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        ge.append(g)
        l_bars.append(Bar(0, 20))
        r_bars.append(Bar(WINDOW_WIDTH - Bar.WIDTH, -20))
        balls.append(Ball())
    
    run = True
    clock = pygame.time.Clock()

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit() 

        # Check for collision and give rewards

        for x, b in enumerate(balls):
            b.move()

            if b.collision(l_bars[x], r_bars[x]):
                ge[x].fitness += 100

            if b.check_win() == None:
                ge[x].fitness += 0.1
            else:
                ge[x].fitness -= 100
                ge.pop(x)
                nets.pop(x)
                balls.pop(x)
                l_bars.pop(x)
                r_bars.pop(x)
                spec -= 1

        # Break if all genomes extinct
        if len(balls) <= 0:
            run = False
            break

        # Let AI decide what to do
        for x, b in enumerate(balls):
            output = nets[0].activate((l_bars[x].center() - balls[x].y, balls[x].angle, balls[x].y, balls[x].going_left(), r_bars[x].center() - balls[x].y))

            l_bars[x].move(output[0])
            r_bars[x].move(output[0])
            
        
        # Print on screen
        bg.fitness = round(ge[0].fitness)
        bg.gen = gen
        bg.species = spec

        game_refresh(GAME_WINDOW, bg, l_bars[0], r_bars[0], balls[0])


def run(config_path):
    # Set configuration file
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Set population
    p = neat.Population(config)

    # Add reported for stats
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Set the fitness function
    winner = p.run(eval_genomes, 1000)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
