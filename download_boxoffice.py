import requests
import os

# =========================================================
# BOX OFFICE DATASET DOWNLOADER
# =========================================================

# GitHub raw CSV URL
URL = (
    "https://raw.githubusercontent.com/"
    "jainharsh524/Bollywood_Movies_Metadata-Cinelytics/"
    "main/Dataset_with_Verdict.csv"
)

# Output location
OUTPUT_FILE = os.path.join(
    "dataset",
    "boxoffice_raw.csv"
)


# =========================================================
# CREATE DATASET FOLDER
# =========================================================

os.makedirs("dataset", exist_ok=True)


# =========================================================
# DOWNLOAD DATASET
# =========================================================

print()
print("==============================================")
print("      MOVIE BOX OFFICE DATASET DOWNLOADER")
print("==============================================")
print()

print("Downloading dataset...")
print("Please wait...")
print()


try:

    response = requests.get(
        URL,
        timeout=120
    )

    # Check whether download was successful
    if response.status_code == 200:

        # Save CSV file
        with open(
            OUTPUT_FILE,
            "wb"
        ) as file:

            file.write(
                response.content
            )

        print()
        print("==============================================")
        print("DOWNLOAD SUCCESSFUL")
        print("==============================================")
        print()
        print("Dataset saved at:")
        print(
            os.path.abspath(
                OUTPUT_FILE
            )
        )
        print()
        print(
            "File size:",
            round(
                len(response.content) / (1024 * 1024),
                2
            ),
            "MB"
        )
        print()
        print("Next step:")
        print("python prepare_boxoffice.py")
        print()

    else:

        print()
        print("==============================================")
        print("DOWNLOAD FAILED")
        print("==============================================")
        print()

        print(
            "HTTP Status Code:",
            response.status_code
        )

        print()
        print(
            "The GitHub dataset URL may have changed."
        )


except requests.exceptions.Timeout:

    print()
    print("==============================================")
    print("DOWNLOAD TIMEOUT")
    print("==============================================")
    print()

    print(
        "The download took too long."
    )

    print(
        "Please check your internet connection and try again."
    )


except requests.exceptions.ConnectionError:

    print()
    print("==============================================")
    print("CONNECTION ERROR")
    print("==============================================")
    print()

    print(
        "Could not connect to GitHub."
    )

    print(
        "Please check your internet connection."
    )


except Exception as error:

    print()
    print("==============================================")
    print("ERROR")
    print("==============================================")
    print()

    print(
        "Something went wrong:"
    )

    print(error)
