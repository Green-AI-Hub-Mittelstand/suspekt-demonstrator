import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="System180 demonstrator launcher")
    parser.add_argument("--mode", choices=["normal", "game"], default="normal")
    args = parser.parse_args()

    if args.mode == "game":
        from demonstrator.apps import game

        game.main()
        return

    from demonstrator.apps import normal

    normal.main()


if __name__ == "__main__":
    main()
