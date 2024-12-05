import cv2 as cv
import numpy as np
import tensorflow as tf
import os
import time
import itertools
from collections import Counter

# Initialize camera
cam = cv.VideoCapture(0)

class Player:
    def __init__(self, chips):
        self.chips = chips
        self.current_bet = 0
        self.is_active = True
        self.is_check = False

    def place_bet(self, amount):
        if amount > self.chips:
            raise ValueError(f"{self.name} doesn't have enough chips to bet {amount}.")
        self.chips -= amount
        self.current_bet += amount

    def check(self):
        self.is_check = True
    
    def fold(self):
        self.is_active = False

    def reset_bet(self):
        self.current_bet = 0


class PokerGame:
    def __init__(self, player1, player2):
        self.players = [player1, player2]
        self.pot = 0
        self.current_bet = 0  # The minimum amount a player must match to stay in

    def add_to_pot(self, amount):
        self.pot += amount
        
class PokerHand:
    CARD_VALUES = {
        'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 
        'eight': 8, 'nine': 9, 'ten': 10, 'jack': 11, 'queen': 12, 'king': 13, 'ace': 14
    }
    
    HAND_RANKINGS = [
        'High Card', 'Pair', 'Two Pair', 'Three of a Kind', 
        'Straight', 'Flush', 'Full House', 'Four of a Kind', 
        'Straight Flush', 'Royal Flush'
    ]

    @staticmethod
    def parse_card(card):
        if isinstance(card, list):
            card = card[0] if card else 'two of hearts'
        value, suit = card.split(' of ')
        return value, suit

    @staticmethod
    def get_card_values(hand):
        return [PokerHand.CARD_VALUES[PokerHand.parse_card(card)[0]] for card in hand]

    @staticmethod
    def get_card_suits(hand):
        return [PokerHand.parse_card(card)[1] for card in hand]

    @classmethod
    def evaluate_hand(cls, hand):
        values = cls.get_card_values(hand)
        suits = cls.get_card_suits(hand)
        
        is_flush = len(set(suits)) == 1
        
        # Check for Straight
        sorted_values = sorted(set(values))
        is_straight = (len(sorted_values) == 5 and sorted_values[-1] - sorted_values[0] == 4) or \
                      (sorted_values == [2, 3, 4, 5, 14])  # Special case for A-2-3-4-5 straight
        
        # Count card occurrences
        value_counts = Counter(values)
        
        # Determine hand ranking
        if is_flush and is_straight:
            # Check for Royal Flush
            if set(values) == {10, 11, 12, 13, 14}:
                return 'Royal Flush', max(values)
            return 'Straight Flush', max(values)
        
        if 4 in value_counts.values():
            return 'Four of a Kind', max(v for v, count in value_counts.items() if count == 4)
        
        if 3 in value_counts.values() and 2 in value_counts.values():
            return 'Full House', max(v for v, count in value_counts.items() if count == 3)
        
        if is_flush:
            return 'Flush', max(values)
        
        if is_straight:
            return 'Straight', max(values)
        
        if 3 in value_counts.values():
            return 'Three of a Kind', max(v for v, count in value_counts.items() if count == 3)
        
        pairs = [v for v, count in value_counts.items() if count == 2]
        if len(pairs) == 2:
            return 'Two Pair', max(pairs)
        
        if len(pairs) == 1:
            return 'Pair', pairs[0]
        
        return 'High Card', max(values)

def determine_winner(player1_hand, player2_hand, table_cards):
    player1_hand = [card['class_name'] if isinstance(card, dict) else card for card in player1_hand]
    player2_hand = [card['class_name'] if isinstance(card, dict) else card for card in player2_hand]
    table_cards = [card['class_name'] if isinstance(card, dict) else card for card in table_cards]
    
    # Combine each player's hand with community cards
    player1_total_hand = player1_hand + table_cards
    player2_total_hand = player2_hand + table_cards
    
    # Get all possible 5-card combinations for each player
    player1_combinations = list(itertools.combinations(player1_total_hand, 5))
    player2_combinations = list(itertools.combinations(player2_total_hand, 5))
    
    # Find the best 5-card hand for each player
    player1_best_rank = max(
        (PokerHand.evaluate_hand(combo) for combo in player1_combinations), 
        key=lambda x: (PokerHand.HAND_RANKINGS.index(x[0]), x[1])
    )
    
    player2_best_rank = max(
        (PokerHand.evaluate_hand(combo) for combo in player2_combinations), 
        key=lambda x: (PokerHand.HAND_RANKINGS.index(x[0]), x[1])
    )
    
    # Determine winner
    player1_rank_index = PokerHand.HAND_RANKINGS.index(player1_best_rank[0])
    player2_rank_index = PokerHand.HAND_RANKINGS.index(player2_best_rank[0])
    
    if player1_rank_index > player2_rank_index:
        return "Player 1", player1_best_rank[0], player2_best_rank[0]
    elif player1_rank_index < player2_rank_index:
        return "Player 2", player1_best_rank[0], player2_best_rank[0]
    else:
        # If ranks are the same, compare the highest card
        if player1_best_rank[1] > player2_best_rank[1]:
            return "Player 1", player1_best_rank[0], player2_best_rank[0]
        elif player1_best_rank[1] < player2_best_rank[1]:
            return "Player 2", player1_best_rank[0], player2_best_rank[0]
        else:
            return "Tie", player1_best_rank[0], player2_best_rank[0]


player1 = Player(1000)
player2 = Player(1000)
game = PokerGame(player1, player2)

# Define states
MENU = "menu"
GAME = "game"
EXIT = "exit"
DATASET = "data"

button_position = (132, 883, 132+293, 883+57)  # x1, y1, x2, y2
current_state = MENU  # Start in the menu state

menu_frame = np.zeros((1444, 2040, 3), dtype=np.uint8)

game_frame = np.zeros((1444, 2040, 3), dtype=np.uint8)
bw = 293
bh = 57

buttons = [
    {"text": "Check", "position": (132, 883, 132+bw, 883+bh), "action": lambda: setattr(player1, 'is_check', True)},
    {"text": "Raise", "position": (132, 992, 132+bw, 992+bh), "action": lambda: player1.place_bet(50)},
    {"text": "Fold", "position": (132, 1101, 132+bw, 1102+bh), "action": lambda: setattr(player1, 'is_active', False)},
    {"text": "Check", "position": (1614, 883, 1614+bw, 883+bh), "action": lambda: setattr(player2, 'is_check', True)},
    {"text": "Raise", "position": (1614, 992, 1614+bw, 992+bh), "action": lambda: player2.place_bet(50)},
    {"text": "Fold", "position": (1614, 1101, 1614+bw, 1102+bh), "action": lambda: setattr(player2, 'is_active', False)},
]

def mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        for button in buttons:
            x1, y1, x2, y2 = button["position"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                button["action"]()  # Execute the associated action

cv.namedWindow("Window")
cv.setMouseCallback("Window", mouse_click)

#Batas Contour Kartu
MIN_AREA = 8500
MAX_AREA = 12000
ASPECT_RATIO = 0.7

#GREEN-SCREEN
lower = np.array([60 - 20, 40, 40])
upper = np.array([60 + 20, 255, 255])

#Trained Model
def load_class_mapping(mapping_file):
    if os.path.exists(mapping_file):
        class_names = {}
        with open(mapping_file, 'r') as f:
            for line in f:
                index, name = line.strip().split(': ')
                class_names[int(index)] = name
        return class_names
    else:
        print(f"Mapping file '{mapping_file}' not found.")
        return None
    
model = tf.keras.models.load_model('model/cnn_model.h5')
class_mapping_file = 'class_mapping.txt'
class_names = load_class_mapping(class_mapping_file)
menu_bg_path = "assets/menu-bg.jpg"
game_bg_path = "assets/new-game-background.png"

def calculate_card_value(card_name):
    parts = card_name.split()
    # Handle face cards and number cards
    if parts[0] == 'ace':
        return 11  # Or 1, depending on Blackjack rules
    elif parts[0] in ['jack', 'queen', 'king']:
        return 10
    else:
        word_to_num_map = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10
        }
        return word_to_num_map.get(parts[0].lower(), -999)
    
def get_card_image_path(class_name, base_dir='datasets-kartu'):
    card_image_dir = os.path.join(base_dir, class_name)
    card_image_file = 'QS_0.jpg'  # Adjust based on your file naming convention
    card_image_path = os.path.join(card_image_dir, card_image_file)
    if os.path.exists(card_image_path):
        return card_image_path
    else:
        print(f"Card image not found for {class_name}. Expected at: {card_image_path}")
        return None

stay = False
temp = False
game_state = "player"
restart = False
loser = None
winner = None
constant_player_score = 0
constant_dealer_score = 0
saved_player_cards = []
saved_dealer_cards = []
saved_deck1_cards = []
fourth_cards = []
final_card = []
final_table_cards = []
pot = 0

def Masking(image):
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    mask = cv.bitwise_not(mask)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv.erode(mask, kernel, iterations=4)
    mask = cv.dilate(mask, kernel, iterations=4)
    return mask

def DetectCards(image):
    global game_state, loser, winner,stay, constant_player_score, constant_dealer_score,restart,saved_dealer_cards,saved_player_cards, saved_deck1_cards, fourth_cards, final_card, final_table_cards, pot
    mask = Masking(image)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    warped_cards = []
    detected_cards = []
    player_score = 0
    dealer_score = 0

    for contour in contours:
        area = cv.contourArea(contour)
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / h
        if MIN_AREA < area < MAX_AREA and 0.65 < aspect_ratio < 0.75:
            continue
        
        epsilon = 0.02 * cv.arcLength(contour,True);
        approx = cv.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            card_points_list = []
            
            for point in approx:
                x, y = point[0][0], point[0][1]
                card_points_list.append((x, y))
                
            card_points = np.array(card_points_list, dtype='float32')
            
            card_points = sorted(card_points, key=lambda y: y[1]) # Sort Y dari kecil ke besar
            top_pts = sorted(card_points[:2], key=lambda x: x[0]) # Sort X dari Koor Y min
            bottom_pts = sorted(card_points[2:], key=lambda x: x[0]) # Sort X dari Koor Y max
            
            ordered_card_points = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype='float32')
            
            top_left = ordered_card_points[0]
            top_right = ordered_card_points[1]
            bottom_right = ordered_card_points[2]
            bottom_left = ordered_card_points[3]
            
            width = int(top_right[0] - top_left[0])
            height = int(bottom_left[1] - top_left[1])
            pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
            
            homography_matrix = cv.getPerspectiveTransform(ordered_card_points, pts_dst)
            warped_card = cv.warpPerspective(image, homography_matrix, (width, height))
            warped_cards.append(warped_card)
            
    for warped_card in warped_cards:
        processed_card = cv.resize(warped_card, (128, 128),interpolation=cv.INTER_LANCZOS4)
        processed_card = processed_card / 255.0 
        processed_card = np.expand_dims(processed_card, axis=0)
        
        prediction = model.predict(processed_card)
        predicted_class= np.argmax(prediction, axis=1)[0]
        class_name = class_names.get(predicted_class, "Unknown") if class_names else "Unknown"
        
        # cv.rectangle(image, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])), (255, 0, 0), 2) 
        # cv.putText(image, f'Class: {class_name}', (x, y), cv.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)
        
        detected_cards.append({
            'class_name': class_name,
            'location': (int(top_left[0]), int(top_left[1]), int(bottom_right[0]), int(bottom_right[1]))
        })
        
    for contour in contours:
        cv.drawContours(image, [contour], -1, (255, 0, 255), 3)
    
    game_background = cv.imread(game_bg_path,cv.IMREAD_COLOR)
    game_background =cv.resize(game_background,(game_frame.shape[1],game_frame.shape[0]))
    game_frame[:] = game_background
    
    cv.putText(game_frame,f'{player1.chips}',(267,1261+50),cv.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),4)
    cv.putText(game_frame,f'{player2.chips}',(1749,1261+50),cv.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),4)
    cv.putText(game_frame,f'{pot}',(1005,1119+50),cv.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),4)
        
    x_start, y_start, x_dealer = 209, 252, 1691  # Starting position for the first card
    gap = 116  # Initial gap between cards
    card_width, card_height = 139, 177  # Card dimensions
    
    dx_start, dy_start =  545, 823
    dgap = 38
        
    if game_state == "player":
        pot = 0
        player1.is_active = True
        player2.is_active = True
        player1.current_bet = 0
        player2.current_bet = 0
        
        for index, card in enumerate(detected_cards):
            class_name = card['class_name']
            y_offset = y_start + (index * (card_height + gap))
            x_offset = x_start

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            if game_frame is not None and game_frame.shape[0] > 0 and game_frame.shape[1] > 0:
                # Ensure the ROI does not exceed the bounds of the game frame
                if y_offset + card_height <= game_frame.shape[0] and x_offset + card_width <= game_frame.shape[1]:
                    roi = game_frame[y_offset:y_offset + card_height, x_offset:x_offset + card_width]
                    if roi.shape == card_image.shape:  # Check if shapes match
                        roi[:] = card_image
                    else:
                        print(f"Shape mismatch at index {index}: ROI shape {roi.shape}, Card shape {card_image.shape}")
                else:
                    print(f"ROI exceeds game frame bounds at index {index}: y_offset {y_offset}, x_offset {x_offset}")
                
        if stay == True:
            saved_player_cards = detected_cards.copy()
            if len(saved_player_cards) == 2:
                game_state = "dealer"
            stay = not stay
        
    elif game_state == "dealer":
        for index, card in enumerate(detected_cards):
            class_name = card['class_name']
            y_offset = y_start + (index * (card_height + gap))
            x_offset = x_dealer

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            if game_frame is not None and game_frame.shape[0] > 0 and game_frame.shape[1] > 0:
                # Ensure the ROI does not exceed the bounds of the game frame
                if y_offset + card_height <= game_frame.shape[0] and x_offset + card_width <= game_frame.shape[1]:
                    roi = game_frame[y_offset:y_offset + card_height, x_offset:x_offset + card_width]
                    if roi.shape == card_image.shape:  # Check if shapes match
                        roi[:] = card_image
                    else:
                        print(f"Shape mismatch at index {index}: ROI shape {roi.shape}, Card shape {card_image.shape}")
                else:
                    print(f"ROI exceeds game frame bounds at index {index}: y_offset {y_offset}, x_offset {x_offset}")
            
        for index, card in enumerate(saved_player_cards):
            class_name = card['class_name']
            y_offset = y_start + index * (card_height + gap) 
            x_offset = x_start

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        if stay == True:
            saved_dealer_cards = detected_cards.copy()
            if len(saved_dealer_cards) == 2:  
                game_state = "draw_1"
            stay = not stay
            
    elif game_state == "draw_1":
        for index, card in enumerate(saved_player_cards):
            class_name = card['class_name']
            y_offset = y_start + index * (card_height + gap)  # Calculate x position with gap
            x_offset = x_start  # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index, card in enumerate(saved_dealer_cards):
            class_name = card['class_name']
            y_offset = y_start + index * (card_height + gap)  # Calculate x position with gap
            x_offset = x_dealer  # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image

        for index, card in enumerate(detected_cards):
            class_name = card['class_name']
            x_offset = dx_start + index * (card_width + dgap)  # Calculate x position with gap
            y_offset = dy_start # Keep y position constant
            
            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found
            
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))
            
            if game_frame.size == 0 or game_frame.shape[0] == 0:
                print("Empty frame detected!")
                continue

            if game_frame is not None and game_frame.shape[0] > 0 and game_frame.shape[1] > 0:
                roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
                if roi.size > 0:
                    roi[:] = card_image
                else:
                    print(f"Invalid ROI at index {index}")
        
        if stay == True:
            player1.is_check = False
            player2.is_check = False
            saved_deck1_cards = detected_cards.copy()
            if len(saved_deck1_cards) == 3:
                game_state = "bet"
            restart = False
            stay = not stay
            
    elif game_state == "bet":
        pot = player1.current_bet + player2.current_bet
        
        for index, card in enumerate(saved_player_cards):
            class_name = card['class_name']
            y_offset = y_start + index * (card_height + gap)  # Calculate x position with gap
            x_offset = x_start  # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index, card in enumerate(saved_dealer_cards):
            class_name = card['class_name']
            y_offset = y_start + index * (card_height + gap)  # Calculate x position with gap
            x_offset = x_dealer  # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index, card in enumerate(saved_deck1_cards):
            class_name = card['class_name']
            x_offset = dx_start + index * (card_height + dgap)  # Calculate x position with gap
            y_offset = dy_start # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        # Draw each button
        for button in buttons:
            x1, y1, x2, y2 = button["position"]
            cv.rectangle(game_frame, (x1, y1), (x2, y2), (50, 200, 50), -1)  # Button color
            cv.putText(
                game_frame,
                button["text"],
                (x1 + 10, y1 + 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        
        if (player1.is_check and player2.is_check == True) and (player1.current_bet == player2.current_bet):
            game_state = "draw_2"
        
        if player1.is_active == False:
            player2.chips += pot
            pot = 0
            restart = True
        
        elif player2.is_active == False:
            player1.chips += pot
            pot = 0
            restart = True
        
        if restart == True:
            game_state = "player"
            loser = None
            winner = None
            restart = False
        
    elif game_state == "draw_2":
        for index, card in enumerate(saved_player_cards):
            class_name = card['class_name']
            y_offset = y_start + index * (card_height + gap)  # Calculate x position with gap
            x_offset = x_start  # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index, card in enumerate(saved_dealer_cards):
            class_name = card['class_name']
            y_offset = y_start + index * (card_height + gap)  # Calculate x position with gap
            x_offset = x_dealer  # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index, card in enumerate(saved_deck1_cards):
            class_name = card['class_name']
            x_offset = dx_start + index * (card_height + dgap)  # Calculate x position with gap
            y_offset = dy_start # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index,card in enumerate(detected_cards):
            class_name = card['class_name']
            x_offset = 1147 + dgap
            y_offset = 823
            
            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
            
        if stay == True:
            player1.is_check = False
            player2.is_check = False
            fourth_cards = detected_cards.copy()
            if len (fourth_cards) == 1:
                game_state = "bet2"
            restart = False
            stay = not stay
        
    elif game_state == "bet2":
        pot = player1.current_bet + player2.current_bet
                
        for index, card in enumerate(saved_player_cards):
            class_name = card['class_name']
            y_offset = y_start + index * (card_height + gap)  # Calculate x position with gap
            x_offset = x_start  # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index, card in enumerate(saved_dealer_cards):
            class_name = card['class_name']
            y_offset = y_start + index * (card_height + gap)  # Calculate x position with gap
            x_offset = x_dealer  # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index, card in enumerate(saved_deck1_cards):
            class_name = card['class_name']
            x_offset = dx_start + index * (card_height + dgap)  # Calculate x position with gap
            y_offset = dy_start # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index,card in enumerate(fourth_cards):
            class_name = card['class_name']
            x_offset = dx_start + 3 * (card_height + dgap)
            y_offset = 823
            
            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        # Draw each button
        for button in buttons:
            x1, y1, x2, y2 = button["position"]
            cv.rectangle(game_frame, (x1, y1), (x2, y2), (50, 200, 50), -1)  # Button color
            cv.putText(
                game_frame,
                button["text"],
                (x1 + 10, y1 + 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        
        if (player1.is_check and player2.is_check == True) and (player1.current_bet == player2.current_bet):
            game_state = "final_draw"
        
        if player1.is_active == False:
            player2.chips += pot
            pot = 0
            restart = True
            
        elif player2.is_active == False:
            player1.chips += pot
            pot = 0
            restart = True
        
        if restart == True:
            game_state = "player"
            loser = None
            winner = None
            stay = not stay
            restart = False
        
        
        
    elif game_state == "final_draw":
        for index, card in enumerate(saved_player_cards):
            class_name = card['class_name']
            y_offset = y_start + index * (card_height + gap)  # Calculate x position with gap
            x_offset = x_start  # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index, card in enumerate(saved_dealer_cards):
            class_name = card['class_name']
            y_offset = y_start + index * (card_height + gap)  # Calculate x position with gap
            x_offset = x_dealer  # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index, card in enumerate(saved_deck1_cards):
            class_name = card['class_name']
            x_offset = dx_start + index * (card_height + dgap)  # Calculate x position with gap
            y_offset = dy_start # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index,card in enumerate(fourth_cards):
            class_name = card['class_name']
            x_offset = dx_start + 3 * (card_height + dgap)
            y_offset = 823
            
            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
            
        for index,card in enumerate(detected_cards):
            class_name = card['class_name']
            x_offset = dx_start + 4 * (card_height + dgap)
            y_offset = 823
            
            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
            
        if stay == True:
            final_card = detected_cards.copy()
            if len(final_card) == 1:  
                game_state = "result"
            stay = not stay
    
    elif game_state == "result":
        for index, card in enumerate(saved_player_cards):
            class_name = card['class_name']
            y_offset = y_start + index * (card_height + gap)  # Calculate x position with gap
            x_offset = x_start  # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index, card in enumerate(saved_dealer_cards):
            class_name = card['class_name']
            y_offset = y_start + index * (card_height + gap)  # Calculate x position with gap
            x_offset = x_dealer  # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index, card in enumerate(saved_deck1_cards):
            class_name = card['class_name']
            x_offset = dx_start + index * (card_height + dgap)  # Calculate x position with gap
            y_offset = dy_start # Keep y position constant

            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        for index,card in enumerate(fourth_cards):
            class_name = card['class_name']
            x_offset = dx_start + 3 * (card_height + dgap)
            y_offset = dy_start
            
            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
            
        for index,card in enumerate(final_card):
            class_name = card['class_name']
            x_offset = dx_start + 4 * (card_height + dgap)
            y_offset = 823
            
            # Get the card image path
            card_path = get_card_image_path(class_name)
            if card_path is None:
                continue  # Skip if card image not found

            # Load and resize the card image
            card_image = cv.imread(card_path, cv.IMREAD_COLOR)
            card_image = cv.resize(card_image, (card_width, card_height))

            # Place the card on the board
            roi = game_frame[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
            roi[:] = card_image
        
        final_table_cards = saved_deck1_cards + fourth_cards + final_card
        winner, player1_rank, player2_rank = determine_winner(saved_player_cards,saved_dealer_cards,final_table_cards)
        
        print(f"Winner: {winner}")
        print(f"Player 1 Best Rank: {player1_rank}")
        print(f"Player 2 Best Rank: {player2_rank}")
        
        cv.putText(game_frame,f'<- {winner} ->',(1005-175,1300),cv.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),4)
        cv.putText(game_frame,f"{player1_rank}",(150,1000),cv.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),4)
        cv.putText(game_frame,f"{player2_rank}",(1649,1000),cv.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),4)
                
        if winner == "Player 1":
            player1.chips += pot
            pot = 0
        elif winner == "Player 2":
            player2.chips += pot
            pot = 0
        else :
            player1.chips += pot/2
            player2.chips += pot/2
            pot = 0
        
        if restart == True:
            loser = None
            winner = None
            game_state = "player"
            restart = False
        
    return image

while True:
    if current_state == MENU:
        background = cv.imread(menu_bg_path, cv.IMREAD_COLOR)
        background = cv.resize(background, (menu_frame.shape[1], menu_frame.shape[0]))
        menu_frame[:] = background
        cv.imshow("Window", menu_frame)

    elif current_state == GAME:
        ret, roi_frame = cam.read()
        
        if not ret:
            cam.set(cv.CAP_PROP_POS_FRAMES, 0)
            ret, roi_frame = cam.read()
        
        DetectCards(roi_frame)
        roi_frame = cv.resize(roi_frame, (949,470))
        
        x, y = (545,252)
        w, h = (949,470)
        
        game_frame[y:y+h, x:x+w] = roi_frame
        cv.imshow("Window", game_frame)
    
    key = cv.waitKey(1) & 0xFF
    
    if key == ord('1'): 
        current_state = GAME
    elif key == ord('s'):
        stay = not stay
        time.sleep(1)
    elif key == ord('r'):
        restart = not restart
        time.sleep(1)
    elif key == ord('m'):
        current_state = MENU
    elif key == ord('q'):
        current_state = EXIT
        break

cam.release()
cv.destroyAllWindows()
