const fs = require('fs');
const csv = require('csv-parser');
const tf = require('@tensorflow/tfjs');
const rawData = [];
const dataFilePath = 'data/data.csv';

let target = process.argv[2];

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
                dez1: parseInt(item.dez1),
                dez2: parseInt(item.dez2),
                dez3: parseInt(item.dez3),
                dez4: parseInt(item.dez4),
                dez5: parseInt(item.dez5),
                dez6: parseInt(item.dez6),
            })
        })
        .on('end', async () => {
            await predict();
        });
}

const predict = async () => {

    // Build and compile model.
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 6, inputShape: [1] }));

    const learningRate = 0.05
    const sgdOptimizer = tf.train.sgd(learningRate)

    model.compile({ loss: 'meanSquaredError', optimizer: sgdOptimizer });

    let min = rawData[rawData.length - 1].date.getTime();
    let max = rawData[0].date.getTime();

    const rawDataXs = rawData.map((item) => {
        return normalize(item.date.getTime(), min, max);
    });

    const rawDataYs = rawData.map((item) => {
        return [item.dez1, item.dez2, item.dez3, item.dez4, item.dez5, item.dez6]
    });

    //Inputs
    const xs = tf.tensor2d(rawDataXs, [rawDataXs.length, 1]);

    // Outputs
    const ys = tf.tensor2d(rawDataYs, [rawDataYs.length, 6]);

    // Train model with fit().
    await model.fit(xs, ys, { epochs: 250 });

    let targetSplit = target.split('/');
    let targetDate = normalize((new Date(targetSplit[2], targetSplit[1], targetSplit[0])).getTime(), min, max);

    // Run inference with predict().
    let prediction = model.predict(tf.tensor2d([target], [1, 1]));
    const readable = prediction.dataSync();
    prediction.print();
}

(async () => {
    await parseFile(dataFilePath);
})();
