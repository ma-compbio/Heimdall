import hydra

from Heimdall.trainer import setup_trainer


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(config):
    trainer = setup_trainer(config)
    if trainer is not None:
        trainer.fit()


if __name__ == "__main__":
    main()
