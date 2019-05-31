const fs = require('fs');
const csv = require('csv-parser');
const tf = require('@tensorflow/tfjs');
const rawData = [];
const dataFilePath = 'data/data.csv';

let target = process.argv[2];
let targetNumber = process.argv[3];
let targetSplit = target.split('/');
let targetDate = new Date(targetSplit[2], targetSplit[1], targetSplit[0]);

const parseDate = (date) => {
    let dateSplit = date.split('/');
    return new Date(dateSplit[2], dateSplit[1], dateSplit[0]);
}

const normalize = (x, min, max) => {
    return (x - min) / (max - min);
}

const parseFile = async (path) => {
    fs.createReadStream(path)
        .pipe(csv())
        .on('data', (item) => {
            rawData.push({
                date: parseDate(item.date),
                value: parseInt(item[`dez${targetNumber}`])
            });
        })
        .on('end', async () => {
            await predict();
        });
}

const predict = async () => {

    // Build and compile model.
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    const learningRate = 0.05
    const sgdOptimizer = tf.train.sgd(learningRate)
    model.compile({ loss: 'meanSquaredError', optimizer: sgdOptimizer });

    let min = rawData[rawData.length - 1].date.getTime();
    let max = targetDate.getTime();

    const rawDataXs = rawData.map((item) => {
        return [normalize(item.date.getTime(), min, max)];
    });

    const rawDataYs = rawData.map((item) => {
        return [item.value]
    });

    //Inputs
    const xs = tf.tensor2d(rawDataXs, [rawDataXs.length, 1]);

    // Outputs
    const ys = tf.tensor2d(rawDataYs, [rawDataYs.length, 1]);

    // Train model with fit().
    await model.fit(xs, ys, { epochs: 300 });

    // Run inference with predict().
    let targetValue = normalize(targetDate.getTime(), min, max);
    let prediction = model.predict(tf.tensor2d([targetValue], [1, 1]));
    //const readable = prediction.dataSync();
    prediction.print();
}

(async () => {
    await parseFile(dataFilePath);
})();
