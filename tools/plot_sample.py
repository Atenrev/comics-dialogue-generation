#import libraries
import argparse
import cv2
from matplotlib import pyplot as plt


CORRECT_COLOR = (0.0, 1.0, 0.0)
INCORRECT_COLOR = (0.0, 0.0, 1.0)
GENERATED_COLOR = (0.0, 0.0, 0.0)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Plotting sample')

    return parser.parse_args()


def main(args) -> None:
    # create figure
    fig = plt.figure(figsize=(14, 7))

    # setting values to rows and column variables
    rows = 2
    columns = 4

    # reading images
    Image1 = cv2.imread('jonas.JPG')
    Image2 = cv2.imread('jonas.JPG')
    Image3 = cv2.imread('jonas.JPG')
    Image4 = cv2.imread('jonas.JPG')

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(Image1)
    plt.axis('off')
    plt.title("Context panel 1")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(Image2)
    plt.axis('off')
    plt.title("Context panel 2")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)

    # showing image
    plt.imshow(Image3)
    plt.axis('off')
    plt.title("Context panel 3")

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, columns, 4)

    # showing image
    plt.imshow(Image4)
    plt.axis('off')
    plt.title("Answer panel")

    # Adding a subplot at the 5th to 7th position
    for i in range(1, 4):
        fig.add_subplot(rows, columns, i+4)

        # showing text
        color = CORRECT_COLOR if i == 1 else INCORRECT_COLOR
        bb = dict(facecolor='white', alpha=1.) if i == 1 else None
        plt.text(0.5, 0.5, f"Candidate {i}", fontsize=20,
                 ha="center", va="center", color=color, bbox=bb)
        plt.axis('off')

    # Adding a subplot at the 8th position
    fig.add_subplot(rows, columns, 8)

    # showing text
    plt.text(0.5, 0.5, "Generated dialogue", fontsize=20,
             ha="center", va="center", color=GENERATED_COLOR)
    plt.axis('off')

    # save the figure
    plt.savefig('sample_plot.png')


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
