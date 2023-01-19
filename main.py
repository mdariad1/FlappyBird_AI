import pygame
import neat
import os
import time
import random

from classes.Base import Base
from classes.Bird import Bird
from classes.Pipe import Pipe

pygame.font.init()
from pygame import HWSURFACE, DOUBLEBUF, RESIZABLE

WIN_WIDTH = 500
WIN_HEIGHT = 800

GEN = 0


def load_img(folder,img_name):
    return pygame.transform.scale2x(pygame.image.load(os.path.join(folder, img_name)))


BG_IMG = load_img("imgs", "bg.png")


STAT_FONT = pygame.font.SysFont("comicsans", 50)


def draw_window(win, birds, pipes, base, score, gen):
    win.blit(BG_IMG, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(text, (10, 10))

    base.draw(win)

    for bird in birds:
        bird.draw(win)
    pygame.display.update()


def main(genomes, config):
    global GEN
    GEN += 1

    checkpoint = 10
    coefficient = 1


    nets = []
    ge = []
    birds = []

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        genome.fitness = 0
        ge.append(genome)

    base = Base(730)
    pipes = [Pipe(700)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT), HWSURFACE | DOUBLEBUF | RESIZABLE)
    clock = pygame.time.Clock()
    score = 0

    run = True
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_index = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_index = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            bird_height = bird.y
            bird_toppipe = abs(bird.y - pipes[pipe_index].height)
            bird_bottompipe = abs(bird.y - pipes[pipe_index].bottom)
            game_speed = pipes[0].VEL * coefficient
            output = nets[x].activate((bird_height, bird_toppipe, bird_bottompipe,game_speed))
            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            for genome in ge:
                genome.fitness += 5
            if score == checkpoint:
                checkpoint += 10
                coefficient += 0.1
            pipe = Pipe(600)
            pipe.VEL *= coefficient
            pipes.append(pipe)


        for r in rem:
            pipes.remove(r)
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        # if score > 25:
        #     for pipe in pipes:
        #         pipe.VEL =
        #     break

        base.move()
        draw_window(win, birds, pipes, base, score, GEN)



def run(config_path):
    config = neat.config.Config(neat.DefaultGenome,neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)