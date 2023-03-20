import argparse
import yaml
import torch

from models import Model
from data import get_test_dataset, get_dataloader

def test(args, cfg):
    device = torch.device('cuda' if cfg['device'] == 'cuda' and torch.cuda.is_available() else 'cpu')
    test_cfg = cfg['test']
    batch_size = test_cfg['batch_size']
    model_cfg = cfg['model']

    # Load the model
    model = Model(model_cfg).to(device)
    checkpoint = torch.load(args.pretrained)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load the test dataset
    test_dataset = get_test_dataset()
    test_loader = get_dataloader(test_dataset, batch_size)

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='Path to config file')
    parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained model file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    test(args, cfg)
