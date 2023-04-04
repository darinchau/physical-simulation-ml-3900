from train import do_stuff, predict_with_name, predict
from load_mesh import load_data, make_anim

if __name__ == "__main__":
    a = predict_with_name("model_1")
    print(a.shape)
    make_anim(a)

    data, _ = load_data()
    make_anim(data)