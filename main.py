import argparse
import configparser

from scripts import convert_data, train_model, visualize_data, evaluate_model, convert_events_to_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run different scripts based on the provided arguments"
    )
    subparsers = parser.add_subparsers(dest="script_name", required=True,
                                       help="Script to run")

    parser_convert_data = subparsers.add_parser("convert_data", help="Convert data")

    parser_train_model = subparsers.add_parser("train_model", help="Train the model")
    parser_train_model.add_argument(
        '--config', type=str, required=True, help='Path to .ini config file for training'
    )

    parser_visualize_data = subparsers.add_parser("visualize_data", help="Visualize data")
    parser_visualize_data.add_argument(
        '--num_samples', type=int, required=True, help='Number of samples to visualize'
    )

    parser_evaluate_model = subparsers.add_parser("evaluate_model", help="Evaluate model")
    parser_evaluate_model.add_argument(
        '--model_path', type=str, required=True, help='Path to model weights'
    )
    parser_evaluate_model.add_argument(
        '--output_dir', type=str, required=True, help='Path to output directory'
    )

    parser_convert_events = subparsers.add_parser("convert_events", help="Convert events to images")

    args = parser.parse_args()

    if args.script_name == 'convert_data':
        convert_data()
    elif args.script_name == 'train_model':
        config = configparser.ConfigParser()
        config.read(args.config)
        print(f'Training with config from {args.config}')
        train_model(config)
    elif args.script_name == 'visualize_data':
        visualize_data(num_samples=args.num_samples)
    elif args.script_name == 'evaluate_model':
        evaluate_model(model_path=args.model_path, output_dir=args.output_dir)
    elif args.script_name == 'convert_events':
        convert_events_to_images()

