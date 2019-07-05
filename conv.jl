using CSV
using DataFrames
using Images
using ImageAxes
using Flux
using Flux: onehotbatch, throttle, crossentropy, @epochs, onecold, maxpool
using Base.Iterators: partition

# Load the images from disk.
train_images_df = CSV.File("./train_images.csv_split_aa", datarow=1) |> DataFrame
test_images_df = CSV.File("./test_images.csv_split_aa", datarow=1) |> DataFrame

# Process the images from CSVs into recognizable image types.
function process_images(images_df)
    images_arr = convert(Array, images_df)
    AxisArray(
        Gray{N0f8}.(reshape(images_arr' / 255, 110, 110, :)),
        :h, :w, :i)
end
images_train = process_images(train_images_df)
images_test = process_images(test_images_df)

# Let's take a look at some of the images
images_train[Axis{:i}(6)]

# Load the labels from disk
train_labels_df = CSV.File("./train_labels.csv") |> DataFrame
test_labels_df = CSV.File("./test_labels.csv") |> DataFrame

# Onehot-encode the binary "has-Volcanoes" label
function onehot_targets(labels_df)
    targets = labels_df[Symbol("Number Volcanoes")]
    targets = [isnan(x) ? 0.0 : 1.0 for x = targets]
    onehotbatch(targets, 0:1)
end
targets_train = onehot_targets(train_labels_df)
targets_test = onehot_targets(test_labels_df)
# Examine the training targets
targets_train[:, 1:10]


function prepare_inputs(images, targets, batch_size)
    reshaped_images = reshape(float.(images), (110, 110, 1, :))
    [(reshaped_images[:, :, :, i], targets[:, i])
         for i in partition(1:last(size(images)), batch_size)]
end

train = prepare_inputs(images_train, targets_train, 100)
test = prepare_inputs(images_test, targets_test, 100)

# A convolutional model
cnn = Chain(
    BatchNorm(1),
    Conv((2, 2), 1=>8, relu),
    x -> maxpool(x, (2, 2)),
    BatchNorm(8),
    Conv((2, 2), 8=>16, relu),
    x -> maxpool(x, (2, 2)),
    BatchNorm(16),
    Conv((2, 2), 16=>16, relu),
    x -> maxpool(x, (2, 2)),
    BatchNorm(16),
    Conv((2, 2), 16=>4, relu),
    x -> maxpool(x, (2, 2)),
    BatchNorm(4),
    x -> reshape(x, :, size(x, 4)),
    BatchNorm(100),
    Dense(100, 2),
    BatchNorm(2),
    softmax)
@time cnn(train[1][1])

# Baseline model - a simple linear dense layer
baseline = Chain(
    x -> reshape(x, :, size(x, 4)),
    Dense(110*110, 2),
    BatchNorm(2),
    softmax)
@time baseline(train[1][1])

function train_model(m)
    loss(x, y) = crossentropy(m(x), y)

    mean(x) = sum(x) / length(x)
    accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

    Xt_ = test[1][1]
    Yt_ = test[1][2]
    function callback()
        @show loss(Xt_, Yt_)
        @show accuracy(Xt_, Yt_)
    end

    evalcb = throttle(callback, 100)

    opt = ADAM(params(m), 0.1)
    @epochs 10 Flux.train!(loss, train, opt, cb = evalcb)

    println("Final accuracy: $(accuracy())")
end

train_model(baseline)
