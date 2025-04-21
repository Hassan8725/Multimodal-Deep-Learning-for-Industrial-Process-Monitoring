import torch
import torch.nn as nn

class Generic1DCNN(nn.Module):
    def __init__(self, input_channels, num_classes, n_layers, kernel_size, stride, 
                 padding, hidden_channels, dropout_prob, pooling_type='max', use_batch_norm=False):
        
        super(Generic1DCNN, self).__init__()
        
        self.layers = nn.ModuleList()
        in_channels = input_channels

        for _ in range(n_layers):
            # Add a convolutional layer
            conv_layer = nn.Conv1d(in_channels, hidden_channels, kernel_size, stride, padding=padding)
            self.layers.append(conv_layer)

            # Add ReLU activation
            self.layers.append(nn.ReLU())

            # Add pooling layer based on pooling_type
            if pooling_type == 'max':
                self.layers.append(nn.MaxPool1d(kernel_size=2))
            elif pooling_type == 'avg':
                self.layers.append(nn.AvgPool1d(kernel_size=2))

            # Add batch normalization if requested
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_channels))

            # Update in_channels for the next layer
            in_channels = hidden_channels

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(0, out_features=128)
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        x = self.flatten(x)

        if self.fc1.in_features != x.shape[1]:
            self.fc1 = nn.Linear(x.shape[1], out_features=128).to(next(self.parameters()).device)
        
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    

class MultimodalModel(nn.Module):
    def __init__(self, input_channels, num_classes, 
                 force_n_layers, force_kernel_size, force_stride, force_padding, force_hidden_channels,
                 force_pooling_type, force_use_batch_norm,
                 acc_n_layers, acc_kernel_size, acc_stride, acc_padding, acc_hidden_channels,
                 acc_pooling_type, acc_use_batch_norm, dropout_prob):
        super(MultimodalModel, self).__init__()

        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        # Convolutional layers for the force input
        self.force_conv = nn.Sequential()
        in_channels_force = input_channels
        for i in range(force_n_layers):
            conv_layer = nn.Conv1d(in_channels_force, force_hidden_channels, force_kernel_size, 
                                   force_stride, padding=force_padding)
            self.force_conv.add_module(f'force_conv{i}', conv_layer)
            self.force_conv.add_module(f'force_relu{i}', nn.ReLU())
            if force_pooling_type == 'max':
                self.force_conv.add_module(f'force_maxpool{i}', nn.MaxPool1d(kernel_size=2))
            elif force_pooling_type == 'avg':
                self.force_conv.add_module(f'force_avgpool{i}', nn.AvgPool1d(kernel_size=2))
            if force_use_batch_norm:
                self.force_conv.add_module(f'force_batchnorm{i}', nn.BatchNorm1d(force_hidden_channels))
            in_channels_force = force_hidden_channels

        # Convolutional layers for the acceleration input
        self.acc_conv = nn.Sequential()
        in_channels_acc = input_channels
        for i in range(acc_n_layers):
            conv_layer = nn.Conv1d(in_channels_acc, acc_hidden_channels, acc_kernel_size, 
                                   acc_stride, padding=acc_padding)
            self.acc_conv.add_module(f'acc_conv{i}', conv_layer)
            self.acc_conv.add_module(f'acc_relu{i}', nn.ReLU())
            if acc_pooling_type == 'max':
                self.acc_conv.add_module(f'acc_maxpool{i}', nn.MaxPool1d(kernel_size=2))
            elif acc_pooling_type == 'avg':
                self.acc_conv.add_module(f'acc_avgpool{i}', nn.AvgPool1d(kernel_size=2))
            if acc_use_batch_norm:
                self.acc_conv.add_module(f'acc_batchnorm{i}', nn.BatchNorm1d(acc_hidden_channels))
            in_channels_acc = acc_hidden_channels


        self.fc1 = nn.Linear(0, out_features=128)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, force_input, acc_input):
        # Forward pass for the force input
        force_output = self.force_conv(force_input)
        force_output = force_output.view(force_output.size(0), -1)

        # Forward pass for the acceleration input
        acc_output = self.acc_conv(acc_input)
        acc_output = acc_output.view(acc_output.size(0), -1)

        # Concatenate the outputs of both modalities
        combined_output = torch.cat((force_output, acc_output), dim=1)
        
        if self.fc1.in_features != combined_output.shape[1]:
            self.fc1 = nn.Linear(combined_output.shape[1], out_features=128).to(next(self.parameters()).device)

        x = self.fc1(combined_output)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        final_output = self.fc2(x)

        return final_output



class ACC1DCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ACC1DCNN, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=30, padding=1)
        self.relu_1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(kernel_size=7)
        self.batchnorm_1 = nn.BatchNorm1d(32)
        self.dropout_1 = nn.Dropout(0.5)

        self.conv_2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=30)
        self.relu_2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool1d(kernel_size=7)
        self.batchnorm_2 = nn.BatchNorm1d(16)
        self.dropout_2 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=400, out_features=64)
        self.relu_3 = nn.ReLU()  
        self.dropout_2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.maxpool_1(x)
        x = self.dropout_1(x)
        
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.maxpool_2(x)
        x = self.dropout_2(x)

        x = self.flatten(x)
        #print(x.view(x.size(0), -1).shape[1])
        x = self.fc1(x)
        x = self.relu_3(x)
        x = self.dropout_2(x)
        x = self.fc2(x)
        return x


class Force1DCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Force1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 75, 128)  # 75 is the size after maxpooling
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class MM1DCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(MultimodalModel, self).__init__()

        # Convolutional layers for the force input
        self.force_conv = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Convolutional layers for the acceleration input
        self.acceleration_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=32, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5)
        )

        # Fully connected layers for combining modalities
        self.fc = nn.Sequential(
            nn.Linear(5136, 128),  # Adjust input size accordingly
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output has 4 dimensions
        )

    def forward(self, force_input, acceleration_input):
        # Forward pass for the force input
        force_output = self.force_conv(force_input)
        force_output = force_output.view(force_output.size(0), -1)

        # Forward pass for the acceleration input
        acceleration_output = self.acceleration_conv(acceleration_input)
        acceleration_output = acceleration_output.view(acceleration_output.size(0), -1)

        # Concatenate the outputs of both modalities
        combined_output = torch.cat((force_output, acceleration_output), dim=1)

        #print(combined_output.shape)

        # Fully connected layers for final prediction
        final_output = self.fc(combined_output)

        return final_output