import torch
from torch.optim import Adam
from my_adam import MyAdam

def test_against_torch_adam():
    torch.manual_seed(42)
    
    model1 = torch.nn.Linear(10, 1)
    model2 = torch.nn.Linear(10, 1)
    model2.load_state_dict(model1.state_dict())

    opt1 = MyAdam(model1.parameters(), lr=1e-3)
    opt2 = Adam(model2.parameters(), lr=1e-3)

    x = torch.randn(16, 10)
    y = torch.randn(16, 1)

    loss_fn = torch.nn.MSELoss()

    for _ in range(10):
        # Our Adam
        opt1.zero_grad()
        loss1 = loss_fn(model1(x), y)
        loss1.backward()
        opt1.step()
        
        # Torch Adam
        opt2.zero_grad()
        loss2 = loss_fn(model2(x), y)
        loss2.backward()
        opt2.step()
    
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), "Parameters diverged from torch Adam"
    print("Test passed: MyAdam matches torch.optim.Adam")

def run_tests():
    test_against_torch_adam()
    
if __name__ == "__main__":
    run_tests()
