from torchsummary import summary
import torchvision.models as models
model = models.resnet152()
model = model.cuda()
summary(model, input_size=(3,224,224), batch_size=-1, device='cuda')
