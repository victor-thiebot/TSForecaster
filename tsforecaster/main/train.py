import argparse
from tsforecaster.data.dataset import TimeSeriesDataset
from tsforecaster.data.preprocessor import DataPreprocessor
from tsforecaster.models import load_model
from tsforecaster.utils.config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Forecasting Prediction')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--input', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the predictions')
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config()

    # Load and preprocess the input data
    input_data = TimeSeriesDataset(args.input, config)
    preprocessor = DataPreprocessor(config)
    preprocessed_data = preprocessor.preprocess(input_data)

    # Load the trained model
    model = load_model(args.model)

    # Make predictions
    predictions = model.predict(preprocessed_data)

    # Save the predictions
    predictions.to_csv(args.output, index=False)

    print(f'Predictions saved to {args.output}')


if __name__ == '__main__':
    main()