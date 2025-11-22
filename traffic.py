import pygame
import random
import time
import winsound

# -------------------------------
#  PYGAME INITIAL SETUP
# -------------------------------
pygame.init()
WIDTH, HEIGHT = 700, 500
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Smart Traffic Controller - Pygame")

FONT = pygame.font.SysFont("Arial", 22)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
YELLOW = (200, 200, 0)
GREEN = (0, 200, 0)
GREY = (50, 50, 50)

clock = pygame.time.Clock()


# -------------------------------
#  HELPER FUNCTIONS
# -------------------------------
def play_emergency_sound():
    """Play a siren beep for emergency."""
    for i in range(3):
        winsound.Beep(1200, 200)
        time.sleep(0.1)


def get_vehicle_count():
    """Simulated vehicle detection."""
    return {
        "cars": random.randint(5, 20),
        "bikes": random.randint(5, 20),
        "trucks": random.randint(1, 5)
    }


def calculate_green_time(vehicles):
    total = vehicles["cars"] + vehicles["bikes"] + vehicles["trucks"]
    green = min(max(total * 1.2, 8), 40)
    return int(green)


def emergency_detected():
    return random.randint(1, 18) == 1


# -------------------------------
#  DRAW TRAFFIC SIGNAL
# -------------------------------
def draw_signal(x, y, state):
    """Draw a vertical traffic signal (red/yellow/green)."""
    pygame.draw.rect(WIN, GREY, (x, y, 60, 150), border_radius=10)

    pygame.draw.circle(WIN, RED if state == "RED" else (80, 0, 0), (x + 30, y + 30), 20)
    pygame.draw.circle(WIN, YELLOW if state == "YELLOW" else (80, 80, 0), (x + 30, y + 75), 20)
    pygame.draw.circle(WIN, GREEN if state == "GREEN" else (0, 80, 0), (x + 30, y + 120), 20)


# -------------------------------
#  MAIN LOOP
# -------------------------------
signals = ["North", "East", "South", "West"]
current = 0
green_timer = 10
vehicles = get_vehicle_count()

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Simulate Emergency
    emergency = emergency_detected()

    if emergency:
        green_timer = 6
        play_emergency_sound()
    else:
        green_timer = calculate_green_time(vehicles)

    # Countdown
    for sec in range(green_timer, 0, -1):

        WIN.fill((30, 30, 30))

        # Draw current signal
        draw_signal(320, 150, "GREEN")

        # Display data
        text1 = FONT.render(f"Current Signal: {signals[current]}", True, WHITE)
        text2 = FONT.render(f"Cars: {vehicles['cars']}  Bikes: {vehicles['bikes']}  Trucks: {vehicles['trucks']}", True, WHITE)
        text3 = FONT.render(f"Green Time Left: {sec} sec", True, WHITE)
        text4 = FONT.render("🚨 EMERGENCY MODE" if emergency else "Normal Mode", True, YELLOW)

        WIN.blit(text1, (20, 20))
        WIN.blit(text2, (20, 60))
        WIN.blit(text3, (20, 100))
        WIN.blit(text4, (20, 140))

        pygame.display.update()
        clock.tick(1)

    # After green ends
    vehicles = get_vehicle_count()
    current = (current + 1) % len(signals)

pygame.quit()