import os
import nemo
import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML
from omegaconf import DictConfig
from utils import load_data
from jiwer import wer, cer
import pytorch_lightning as pl

def load_configuration_from_yaml(config_path):
    """
    Load model configuration from a YAML file.
    """
    yaml_loader = YAML(typ='safe')
    with open(config_path, 'r') as file:
        parameters = yaml_loader.load(file)
    return parameters


def setup_model_parameters(parameters):
    parameters['model']['train_ds']['manifest_filepath'] = "train_manifest.json"
    parameters['model']['validation_ds']['manifest_filepath'] = "test_manifest.json"


def compute_word_error_rate(model):
    wer_numerators = []
    wer_denominators = []

    for test_batch in model.test_dataloader():
        # Move test batch to GPU
        test_batch = [x.cuda() for x in test_batch]

        # Extract targets and their lengths
        targets = test_batch[2]
        targets_lengths = test_batch[3]

        # Get model predictions
        log_probs, encoded_len, greedy_predictions = model(
            input_signal=test_batch[0], input_signal_length=test_batch[1]
        )

        # Update WER using model's helper object
        model._wer.update(greedy_predictions, targets, targets_lengths)
        _, wer_num, wer_denom = model._wer.compute()
        model._wer.reset()

        # Store WER numerators and denominators
        wer_numerators.append(wer_num.detach().cpu().numpy())
        wer_denominators.append(wer_denom.detach().cpu().numpy())

        # Release tensors from GPU memory
        del test_batch, log_probs, targets, targets_lengths, encoded_len, greedy_predictions

    return sum(wer_numerators) / sum(wer_denominators)

def test_model():
    params = load_configuration_from_yaml(config_path)
    model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(params['model']), trainer=trainer)
    model = model.restore_from(
        "trained/jasper-model.nemo")
    test_data = load_data("test")
    test_audio_files = [sample["audio_path"] for sample in test_data.values()]
    ground_truth = [sample["label"] for sample in test_data.values()]

    hypothesis = model.transcribe(test_audio_files)

    wer = wer(ground_truth, hypothesis)
    cer = cer(ground_truth, hypothesis)

    print(f"wer: {wer}")
    print(f"cer: {cer}")


if __name__ == "__main__":
    # Set up the trainer
    trainer = pl.Trainer(devices=1, accelerator='gpu', max_epochs=50)

    # Load configuration parameters
    config_path = 'configs/config.yaml'
    parameters = load_configuration_from_yaml(config_path)

    # Set up model parameters
    setup_model_parameters(parameters)

    # Initialize the model
    model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(parameters['model']), trainer=trainer)

    # Train the model
    trainer.fit(model)

    # Set up test data and move model to GPU
    model.setup_test_data(test_data_config=parameters['model']['validation_ds'])
    model.cuda()
    model.eval()

    # Compute and print WER
    wer = compute_word_error_rate(model)
    print(f"WER = {wer}")
