import os
import anthropic
import base64
import cv2
from tqdm import tqdm

def get_text_label(img_path):
    client = anthropic.Anthropic(
        api_key="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",
    )

    image1 = cv2.imread(img_path)
    _, img_encoded = cv2.imencode('.jpg', image1)
    image1_data = base64.b64encode(img_encoded).decode("utf-8")
    image1_media_type = "image/jpeg"

    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image1_media_type,
                            "data": image1_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Which of the following expressions is the expression in the image: 'Happily Surprised', 'Sadly Fearful', 'Sadly Angry', 'Sadly Surprised', 'Fearfully Surprised', 'Angrily Surprised',  'Disgustedly Surprised',No explanation needed, just give me the name of a category"
                    }
                ],
            }
        ],
    )
    return message.content[0].text

# Specify the directory path
directory = '/abaw-test/images_aligned'

# Initialize an empty list to store the image paths
image_list = []

# Recursively iterate over the files in the directory and its subdirectories
for root, dirs, files in os.walk(directory):
    for filename in files:
        # Check if the file is an image file
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Get the full path of the image file
            image_path = os.path.join(root, filename)
            # Add the image path to the list
            image_list.append(image_path)

error_images = []
success_count = 0  

for idx, image_name in tqdm(enumerate(image_list), total=len(image_list), desc='Generating text'):
    text_file_path = os.path.splitext(image_name)[0] + '.txt'
    if os.path.exists(text_file_path) and os.path.getsize(text_file_path) != 0:
        continue
    else:
        try:
            image_path = os.path.join(directory, image_name)
            generated_text = get_text_label(image_path)
            with open(text_file_path, 'w') as file:
                file.write(generated_text)
            success_count += 1  
        except:
            error_images.append(image_name)

print('error length:', len(error_images)) 
