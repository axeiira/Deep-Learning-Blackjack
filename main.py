import cv2 as cv
import numpy as np
import os
import time
import tensorflow as tf

#Batas Contour Kartu
MIN_AREA = 8500
MAX_AREA = 12000
ASPECT_RATIO = 0.7

#GREEN-SCREEN
lower = np.array([60 - 20, 40, 40])
upper = np.array([60 + 20, 255, 255])

#Trained Model
model = tf.keras.models.load_model('model/cnn_model.h5')
menu_bg_path = "assets/menu-bg.jpg"
game_bg_path = "assets/game-window.jpg"

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

class_mapping_file = 'class_mapping.txt'
class_names = load_class_mapping(class_mapping_file)

cam = cv.VideoCapture(0)
last_saved_time = time.time()

def DrawCircle(image, k, b):
    center_coor = (b,k)
    radius = 12
    color = (175, 175, 0)
    thickness = -1
    image = cv.circle(image, center_coor, radius, color, thickness)
    return image

def create_button(image, button_position, button_size, text, text_color=(255, 255, 255), button_color=(255, 0, 0)):
    x, y = button_position
    width, height = button_size

    # Draw the button (rectangle)
    cv.rectangle(image, (x, y), (x + width, y + height), button_color, -1)

    # Set up the text
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    # Get text size to center it on the button
    text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
    text_x = x + (width - text_size[0]) // 2
    text_y = y + (height + text_size[1]) // 2

    # Draw the text
    cv.putText(image, text, (text_x, text_y), font, font_scale, text_color, thickness, cv.LINE_AA)

    return image

def Masking(image):
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower, upper)
    mask = cv.bitwise_not(mask)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv.erode(mask, kernel, iterations=4)
    mask = cv.dilate(mask, kernel, iterations=4)
    return mask

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

def DetectCards(image,state):
    for i in range(10):
        cv.destroyWindow(f"Warped Card {i + 1}")
    global game_state, loser, winner,stay, constant_player_score, constant_dealer_score,restart,saved_dealer_cards,saved_player_cards
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
        if game_state == "player": 
            player_score = sum(calculate_card_value(card['class_name']) for card in detected_cards) 
        elif game_state == "dealer":
            dealer_score = sum(calculate_card_value(card['class_name']) for card in detected_cards) 
        
    for contour in contours:
        cv.drawContours(image, [contour], -1, (255, 0, 255), 3)
    
    if state == 'play': 
        game_background = cv.imread(game_bg_path,cv.IMREAD_COLOR)
        game_background =cv.resize(game_background,(image.shape[1],image.shape[0]))
        board_window = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        board_window[:] = game_background
        
        x_start, y_start, y_dealer = 193, 690, 230  # Starting position for the first card
        gap = 20  # Initial gap between cards
        card_width, card_height = 109, 138  # Card dimensions    
        
        cv.putText(board_window,"Player",(173,960),cv.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
        cv.putText(board_window,"Dealer",(173,460),cv.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)

        
        if game_state == "player":
            for index, card in enumerate(detected_cards):
                class_name = card['class_name']
                x_offset = x_start + index * (card_width + gap)  # Calculate x position with gap
                y_offset = y_start  # Keep y position constant

                # Get the card image path
                card_path = get_card_image_path(class_name)
                if card_path is None:
                    continue  # Skip if card image not found

                # Load and resize the card image
                card_image = cv.imread(card_path, cv.IMREAD_COLOR)
                card_image = cv.resize(card_image, (card_width, card_height))

                # Place the card on the board
                roi = board_window[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
                roi[:] = card_image
                
            if stay == True:
                saved_player_cards = detected_cards.copy()
                if constant_player_score == 0:
                    constant_player_score = player_score
                    stay = not stay
                    
            if stay != True and constant_player_score != 0:
                game_state = "dealer"
            
        elif game_state == "dealer":
            for index, card in enumerate(detected_cards):
                class_name = card['class_name']
                x_offset = x_start + index * (card_width + gap)  # Calculate x position with gap
                y_offset = y_dealer  # Keep y position constant

                # Get the card image path
                card_path = get_card_image_path(class_name)
                if card_path is None:
                    continue  # Skip if card image not found

                # Load and resize the card image
                card_image = cv.imread(card_path, cv.IMREAD_COLOR)
                card_image = cv.resize(card_image, (card_width, card_height))

                # Place the card on the board
                roi = board_window[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
                roi[:] = card_image
                
            for index, card in enumerate(saved_player_cards):
                class_name = card['class_name']
                x_offset = x_start + index * (card_width + gap)  # Calculate x position with gap
                y_offset = y_start  # Keep y position constant

                # Get the card image path
                card_path = get_card_image_path(class_name)
                if card_path is None:
                    continue  # Skip if card image not found

                # Load and resize the card image
                card_image = cv.imread(card_path, cv.IMREAD_COLOR)
                card_image = cv.resize(card_image, (card_width, card_height))

                # Place the card on the board
                roi = board_window[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
                roi[:] = card_image
            
            if stay == True:
                saved_dealer_cards = detected_cards.copy()
                if constant_dealer_score == 0:
                    constant_dealer_score = dealer_score
                game_state = "result"
                stay = not stay
                
        # elif game_state == "bust":
        #     cv.putText(black_image, f'{loser} Bust !!!',(0,20),cv.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2)
        #     cv.putText(black_image, f'{winner} Win',(0,20),cv.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2)
        
        elif game_state == "result":
            # cv.putText(board_window,'Result Of The Match', (80, 150), cv.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2, cv.LINE_AA)
            if constant_dealer_score > 21 or constant_player_score > constant_dealer_score:
                cv.putText(board_window,"Player Wins",(int(board_window.shape[1]/2)-10,int(board_window.shape[0]/2)),cv.FONT_HERSHEY_SIMPLEX,2.0,(255,255,255),2)
            elif constant_player_score < constant_dealer_score:
                cv.putText(board_window,"Dealer Wins",(int(board_window.shape[1]/2)-10,int(board_window.shape[0]/2)),cv.FONT_HERSHEY_SIMPLEX,2.0,(255,255,255),2)
            else:
                cv.putText(board_window,"Draw",(int(board_window.shape[1]/2)-10,int(board_window.shape[0]/2)),cv.FONT_HERSHEY_SIMPLEX,2.0,(255,255,255),2)
            
            for index, card in enumerate(saved_player_cards):
                class_name = card['class_name']
                x_offset = x_start + index * (card_width + gap)  # Calculate x position with gap
                y_offset = y_start  # Keep y position constant

                # Get the card image path
                card_path = get_card_image_path(class_name)
                if card_path is None:
                    continue  # Skip if card image not found

                # Load and resize the card image
                card_image = cv.imread(card_path, cv.IMREAD_COLOR)
                card_image = cv.resize(card_image, (card_width, card_height))

                # Place the card on the board
                roi = board_window[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
                roi[:] = card_image
                
            for index, card in enumerate(saved_dealer_cards):
                class_name = card['class_name']
                x_offset = x_start + index * (card_width + gap)  # Calculate x position with gap
                y_offset = y_dealer  # Keep y position constant

                # Get the card image path
                card_path = get_card_image_path(class_name)
                if card_path is None:
                    continue  # Skip if card image not found

                # Load and resize the card image
                card_image = cv.imread(card_path, cv.IMREAD_COLOR)
                card_image = cv.resize(card_image, (card_width, card_height))

                # Place the card on the board
                roi = board_window[y_offset:y_offset+card_height, x_offset:x_offset+card_width]
                roi[:] = card_image
            
            if restart == True:
                game_state = "player"
                loser = None
                winner = None
                constant_player_score = 0
                constant_dealer_score = 0
                restart = False
            
        return image,board_window
    
    elif state == 'datasets':    
        return image,warped_cards
    
    return image

initial_state = 'menu'
state = initial_state

def Menu(frame):
    background = cv.imread(menu_bg_path, cv.IMREAD_COLOR)
    background = cv.resize(background, (frame.shape[1], frame.shape[0]))
    frame[:] = background
    
def save_images(folder_path, warp_list, counter):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i, warp in enumerate(warp_list):
        filename = f"{folder_path}/QS_{counter}.jpg"
        cv.imwrite(filename, warp)

def Upload_to_Datasets(cam):
    start_time = time.time()
    folder_path = "datasets"  
    interval = 1  # Seconds
    frame_taken = 0
    MAX_FRAME = 64
    is_paused = False
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if not is_paused and elapsed_time >= interval:
            _, warp_list = DetectCards(frame, 'datasets')
            save_images(folder_path, warp_list, frame_taken)
            frame_taken += 1
            start_time = current_time 

        status_text = "Paused" if is_paused else f"Frame taken: {frame_taken}"
        cv.putText(frame, status_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv.imshow('Frame', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_paused = not is_paused
        
        if frame_taken >= MAX_FRAME:
            break

    cam.release()
    cv.destroyAllWindows()
    return 0

    
def handleState(key, state, frame):
    if key == ord('1'):
        state = 'play'
        
    elif key == ord('2'):
        state = 'datasets'
    
    if state == 'menu':
        frame[:] = (255, 255, 255)
        Menu(frame)
        cv.imshow('Menu', frame)
        
    elif state == 'play':
        frame, stats = DetectCards(frame,state)
        cv.imshow('Frame', frame)
        cv.imshow('Status', stats)
        
    elif state == 'datasets':
        Upload_to_Datasets(cam)
        state = 'menu'

    return state

while True:
    ret, image = cam.read()
    
    if not ret:
        print("Failed Frame")
        break
        
    key = cv.waitKey(1) & 0xFF
    state = handleState(key, state, image)
    
    if key == ord('s'):
        stay = not stay
        time.sleep(1)
    
    if key == ord('r'):
        restart = not restart
        time.sleep(1)
    
    if key == ord('3'):
        break
    
cv.destroyAllWindows()