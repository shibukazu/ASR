import requests


def main():
    LINKS = [
        "https://sensix.tech/libriadapt/libriadapt-en-us.tar.gz.part_aa",
        "https://sensix.tech/libriadapt/libriadapt-en-us.tar.gz.part_ab",
        "https://sensix.tech/libriadapt/libriadapt-en-us.tar.gz.part_ac",
        "https://sensix.tech/libriadapt/libriadapt-en-us.tar.gz.part_ad",
        "https://sensix.tech/libriadapt/libriadapt-en-us.tar.gz.part_ae",
        "https://sensix.tech/libriadapt/libriadapt-en-in.tar.gz.part_aa",
        "https://sensix.tech/libriadapt/libriadapt-en-in.tar.gz.part_ab",
        "https://sensix.tech/libriadapt/libriadapt-en-gb.tar.gz.part_aa",
        "https://sensix.tech/libriadapt/libriadapt-en-gb.tar.gz.part_ab",
        "https://sensix.tech/libriadapt/libriadapt-noise.tar.gz",
    ]
    for link in LINKS:
        print("Downloading", link)
        filepath = "../../datasets/libriadapt/" + link.split("/")[-1]
        print("Saving to", filepath)
        r = requests.get(link)
        if r.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(r.content)
        else:
            print("Error", r.status_code)


if __name__ == "__main__":
    main()
