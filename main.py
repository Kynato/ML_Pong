import pygame
import neat
import random
import math
import os

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
BAR_SIZE = 300
BALL_SIZE = 30
FPS = 240
pygame.font.init()

GAME_WINDOW = pygame.display.set_mode( (WINDOW_WIDTH, WINDOW_HEIGHT ))

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

class Ball:
    SIZE = BALL_SIZE
    VEL = 10
    def __init__(self):
        self.x = WINDOW_WIDTH/2 - self.SIZE/2
        self.y = WINDOW_HEIGHT/2 - self.SIZE/2
        self.color = (255, 255, 255)
        self.angle = random.randrange(-30, 30)

    def move(self):
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

    def collision(self, left, right): 

        if self.x <= left.spacing + left.WIDTH:
            if self.y > left.top and self.y < left.bottom:
                # Lewy odbija
                self.angle = 180 - self.angle
                return True

        elif self.x + self.SIZE >= WINDOW_WIDTH - right.WIDTH + right.spacing:
            if self.y > right.top and self.y < right.bottom:
                # Prawy odbija
                self.angle = 180 - self.angle
                return True

        if self.y <= 0 or self.y + self.SIZE >= WINDOW_HEIGHT:
            self.angle = 360 - self.angle


class Background:
    def __init__(self, color=None):
        if not color:
            self.color = (0, 0, 0)
        else:
            self.color = color
        self.fitness = 0
        self.STAT_FONT = pygame.font.SysFont("comicsans", 50)

    def draw(self, win):
        bg = pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(win, self.color, bg)
        # Score
        text = self.STAT_FONT.render("Score: " + str(self.fitness), True, pygame.Color('white'))
        win.blit(text, (WINDOW_WIDTH - 10 - text.get_width(), 10))


def game_refresh(win, bg, lb, rb, ball):

    bg.draw(win)

    lb.draw(win)
    rb.draw(win)

    ball.draw(win)

    pygame.display.update()

bg = Background() 

def eval_genomes(genomes, config):
    nets = []
    ge = []
    print(len(genomes))
    
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        ge.append(g)
    
    if ge[1] != None:
        ge.pop(1)
    if nets[1] != None:
        nets.pop(1)

    lb = Bar(0, 20)
    rb = Bar(WINDOW_WIDTH - Bar.WIDTH, -20)
    ball = Ball()
    run = True
    clock = pygame.time.Clock()

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit() 

        if ball.check_win() != None:
            run = False
            ge[0].fitness -= 10
            
        ge[0].fitness += 0.1
        lmid = lb.top + lb.HEIGHT/2
        rmid = rb.top + rb.HEIGHT/2
        #output = nets[0].activate((lb.top, lb.bottom, ball.x, ball.y, rb.top, rb.bottom))
        output = nets[0].activate((lb.top, lb.bottom, abs(lmid - ball.y), ball.angle, rb.top, rb.bottom, abs(rmid - ball.y)))
        lb.move(output[0])
        rb.move(output[1])
        
        ball.move()
        if ball.collision(lb, rb):
            ge[0].fitness += 10
        bg.fitness = round(ge[0].fitness)

        game_refresh(GAME_WINDOW, bg, lb, rb, ball)


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
