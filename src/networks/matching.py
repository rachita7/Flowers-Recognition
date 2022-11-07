import torch
import torchvision


class MatchingNetwork(torch.nn.Module):
    
    def __init__(self, backbone=None, image_size=224, use_full_contextual_embedding=True) -> None:
        super().__init__()
        
        self.use_full_contextual_embedding = use_full_contextual_embedding
        
        self.backbone = backbone
        
        if self.backbone is None:
            self.backbone = self.get_backbone()
        
        self.feature_size = self.get_output_shape(self.backbone, image_size)[0]
        
        self.support_encoder = torch.nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.feature_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        self.query_encoder = torch.nn.LSTMCell(self.feature_size * 2, self.feature_size)
        
        self.contextualized_support_features = None
        self.one_hot_support_labels = None
        
        self.softmax = torch.nn.Softmax(dim=1)
        
    def get_backbone(self):
        backbone = torchvision.models.resnet18(pretrained=True)
        backbone.fc = torch.nn.Flatten()
        return backbone
                
    def encode_support_set(self, support_images, support_labels):

        support_features = self.backbone(support_images)
        
        if self.use_full_contextual_embedding:
            hidden_state = self.support_encoder(support_features.unsqueeze(0))[0].squeeze(0)
            self.contextualized_support_features = support_features + hidden_state[:, : self.feature_size] + hidden_state[:, self.feature_size :]
            
        else:
            self.contextualized_support_features = support_features

        self.one_hot_support_labels = torch.nn.functional.one_hot(support_labels).float()

    def encode_query_features(self, query_set):
        
        query_features = self.backbone(query_set)

        if not self.use_full_contextual_embedding:
            return query_features
        
        hidden_state = query_features
        cell_state = torch.zeros_like(query_features)

        for _ in range(len(self.contextualized_support_features)):
            attention = self.softmax(
                hidden_state.mm(self.contextualized_support_features.T)
            )
            read_out = attention.mm(self.contextualized_support_features)
            lstm_input = torch.cat((query_features, read_out), 1)

            hidden_state, cell_state = self.query_encoder(
                lstm_input, (hidden_state, cell_state)
            )
            hidden_state = hidden_state + query_features

        return hidden_state
        
    def get_output_shape(self, model, image_size):
        x = torch.randn(1, 3, image_size, image_size)
        out = model(x)
        return out.shape[1:]
    
    def forward(self, support_images, support_labels, query_images):
        
        self.encode_support_set(support_images, support_labels)
        
        contextualized_query_features = self.encode_query_features(
            query_images
        )
        
        similarity_matrix = self.softmax(
            contextualized_query_features.mm(
                torch.nn.functional.normalize(self.contextualized_support_features).T
            )
        )
                
        log_probabilities = (
            similarity_matrix.mm(self.one_hot_support_labels) + 1e-4
        ).log()
        
        return log_probabilities