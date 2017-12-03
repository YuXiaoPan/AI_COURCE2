package me.peyppicp.advance;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * @author YuXiao Pan
 * @date 2017/11/27
 * @email yuxiao.pan@kikatech.com
 */
public class EmojiLSTM {

    private static final Logger log = LoggerFactory.getLogger(EmojiLSTM.class);

    public static void main(String[] args) throws IOException {

//        CudaEnvironment.getInstance().getConfiguration().setMaximumDeviceCache(5L * 1024 * 1024 * 1024);



        int batchSize = 100;
        int nEpochs = 5;
        int truncateReviewsToLength = 300;
        String path = "F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\distinctLines.txt";
        String labelPath = "F:\\WorkSpace\\idea project location\\AI-Emoji\\src\\main\\resources\\commonLabelWithIndex.txt";
        String word2VecPath = "F:\\WorkSpace\\idea project location\\AI-Emoji\\word2vecLookUpTable.txt";
        ExecutorService executorService = Executors.newFixedThreadPool(1);

        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(word2VecPath);

        EDataSetIterator eDataSetIterator = new EDataSetIterator(path, labelPath, wordVectors, batchSize, truncateReviewsToLength, false);
        EDataSetIterator eDataSetIteratorTest = new EDataSetIterator(path, labelPath, wordVectors, batchSize, truncateReviewsToLength, true);
        AsyncDataSetIterator asyncDataSetIterator = new AsyncDataSetIterator(eDataSetIterator);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.RMSPROP)
                .regularization(true)
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .learningRate(0.04)
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(eDataSetIterator.inputColumns()).nOut(250)
                        .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(250).nOut(eDataSetIterator.totalOutcomes()).build())
                .pretrain(false)
                .backprop(true)
                .build();


        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(conf);
        multiLayerNetwork.init();
        int i = 0;
        multiLayerNetwork.setListeners(new ScoreIterationListener(1), new StatsListener(statsStorage), new IterationListener() {
            @Override
            public boolean invoked() {
                return true;
            }

            @Override
            public void invoke() {
            }

            @Override
            public void iterationDone(Model model, int i) {
                log.info("Now is at:" + eDataSetIterator.cursor() + ",total :" + eDataSetIterator.totalExamples());
                i++;
                if (i % 500 == 0) {
//                    executorService.submit(new HibernateRunner(i, model, eDataSetIteratorTest));
                }
            }
        });
        log.info("Starting training");

        for (int j = 0; j < nEpochs; j++) {
            multiLayerNetwork.fit(asyncDataSetIterator);
            eDataSetIterator.reset();
            executorService.submit(new HibernateRunner(j, multiLayerNetwork, eDataSetIteratorTest));
        }
        File file = new File("model-full.txt");
        file.createNewFile();
        ModelSerializer.writeModel(multiLayerNetwork, file, true);

//        Runtime.getRuntime().exec("shutdown -s -t 10");
    }
}

class HibernateRunner implements Runnable {

    private int anInt;
    private String path;
    private MultiLayerNetwork model;
    private DataSetIterator dataSetIterator;

    public HibernateRunner(int anInt, Model model, DataSetIterator iterator) {
        this.dataSetIterator = iterator;
        this.anInt = anInt;
        this.path = "model-" + anInt + ".txt";
        this.model = (MultiLayerNetwork) model;
    }

    @Override
    public void run() {
        try {
//            Evaluation evaluate = model.evaluate(dataSetIterator);
//            System.out.println(evaluate.stats());

            File file = new File(path);
            if (file.exists()) {
                file.delete();
                file.createNewFile();
            }
            file.createNewFile();
            ModelSerializer.writeModel(model, file, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
