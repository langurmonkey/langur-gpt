import torch
import math
from matplotlib import pyplot as plt

N = 14000

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, N)
y = torch.sin(x)

# Prepare the input tensor (x, x^2, x^3, x^4, x^5, x^6, x^7).
p = torch.tensor([1, 2, 3, 4, 5, 6, 7])
xx = x.unsqueeze(-1).pow(p)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(7, 1),
    torch.nn.Flatten(0, 1)
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use RMSprop; the optim package contains many other
# optimization algorithms. The first argument to the RMSprop constructor tells the
# optimizer which Tensors it should update.
learning_rate = 2e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for t in range(N):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(xx)

    # Compute and print loss.
    loss = loss_fn(y_pred, y)
    if t % 99 == 0:
        print(t + 1, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()


linear_layer = model[0]
print(f"""Result: y = {linear_layer.bias.item()}
      + {linear_layer.weight[:, 0].item()} x
      + {linear_layer.weight[:, 1].item()} x^2
      + {linear_layer.weight[:, 2].item()} x^3
      + {linear_layer.weight[:, 3].item()} x^4
      + {linear_layer.weight[:, 4].item()} x^5
      + {linear_layer.weight[:, 5].item()} x^6
      + {linear_layer.weight[:, 6].item()} x^7""")

def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    print(f'# This is a polynomial of order {o}.')
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y


coeffs = [linear_layer.bias.item(),
     linear_layer.weight[:, 0].item(),
     linear_layer.weight[:, 1].item(),
     linear_layer.weight[:, 2].item(),
     linear_layer.weight[:, 3].item(),
     linear_layer.weight[:, 4].item(),
     linear_layer.weight[:, 5].item(),
     linear_layer.weight[:, 6].item()]

plt.plot(x, y, color='#ff9999', linestyle='--', label='Truth')
plt.plot(x, PolyCoefficients(x, coeffs), color='#3355ff', label='Model')
plt.xlabel('Angle (rad)')
plt.ylabel('Value')
plt.title("Model performance")
plt.legend()
plt.show()
