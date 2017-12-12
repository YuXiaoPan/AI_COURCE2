package me.peyppicp.advance2;

import com.google.common.base.Preconditions;
import me.peyppicp.advance.Word2VecMain;
import org.deeplearning4j.api.storage.StatsStorage;
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
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * @author YuXiao Pan
 * @date 2017/11/27
 * @email yuxiao.pan@kikatech.com
 */
public class EmojiLSTM {

    private static final Logger log = LoggerFactory.getLogger(EmojiLSTM.class);
    public static final String OUTPUT = "/home/peyppicp/output/";
//    public static final String PREFIX = "/home/peyppicp/data/new/";
    public static final String PREFIX = "";

    private static void operationFunctionRebuildWordVector(Scanner scanner) throws IOException {
        String[] word2VecArgs = new String[6];
        System.out.println("Please set origin file path:");
        String originPath = scanner.nextLine();
        File file = new File(originPath);
        Preconditions.checkArgument(file.exists());
        word2VecArgs[0] = originPath;
        System.out.println("Please set min word frequency:");
        word2VecArgs[1] = scanner.nextLine();
        System.out.println("Please set iterations:");
        word2VecArgs[2] = scanner.nextLine();
        System.out.println("Please set layerSize:");
        word2VecArgs[3] = scanner.nextLine();
        System.out.println("Please set seed:");
        word2VecArgs[4] = scanner.nextLine();
        System.out.println("Please set window size:");
        word2VecArgs[5] = scanner.nextLine();
        System.out.println("Please set flag to determine weather continue or not: 1->true,0->false");
        int flagInt = scanner.nextInt();
        Preconditions.checkArgument(flagInt == 0 || flagInt == 1);
        Word2VecMain.main(word2VecArgs);
        if (flagInt == 1) {
            System.exit(0);
        }
    }

    private static void operationFunctionRebuildModel(Scanner scanner) throws IOException, Nd4jBackend.NoAvailableBackendException {
//        System.out.println("Please specify prefix for model:");
//        String prefix = scanner.nextLine();
//        System.out.println("Please set train data path:");
//        String trainDataPath = scanner.nextLine();
//        System.out.println("Please set label data path:");
//        String labelDataPath = scanner.nextLine();
//        System.out.println("Please set word vector data path(Please use look up table):");
//        String wordVectorPath = scanner.nextLine();
//        System.out.println("Please set batch size:");
//        int batchSize = scanner.nextInt();
//        System.out.println("Please set nEpochs:");
//        int nEpochs = scanner.nextInt();
//        System.out.println("Please set truncateReviewsToLength:");
//        int truncateReviewsToLength = scanner.nextInt();
//        System.out.println("Please set learning rate(0.00xx):");
//        double learningRate = scanner.nextDouble();

        String wordVectorPath = PREFIX + "LookUpTable.txt";
        String trainDataPath = PREFIX + "EmojiSampleWithoutEmoji.txt";
        String labelDataPath = PREFIX + "EmojiSampleLabels.txt";
        String sampleFilePath = PREFIX + "EmojiSample.txt";
        int batchSize = 200;
        int truncateReviewsToLength = 300;
        double learningRate = 0.01;
        int nEpochs = 200;
        String prefix = "main03";

        ExecutorService executorService = Executors.newFixedThreadPool(1);

        EDataSetIterator eDataSetIterator = new EDataSetIterator(sampleFilePath, trainDataPath, labelDataPath, wordVectorPath, batchSize, truncateReviewsToLength, false);
        EDataSetIterator eDataSetIteratorTest = new EDataSetIterator(sampleFilePath, trainDataPath, labelDataPath, wordVectorPath, batchSize, truncateReviewsToLength, true);
//        AsyncDataSetIterator asyncDataSetIterator = new AsyncDataSetIterator(eDataSetIterator);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(Updater.RMSPROP)
                .regularization(true)
                .l1(1e-4)
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .learningRate(learningRate)
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(eDataSetIterator.inputColumns()).nOut(150)
                        .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(150).nOut(eDataSetIterator.totalOutcomes()).build())
                .pretrain(false)
                .backprop(true)
                .build();


        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

//        NativeOpsHolder.getInstance().getDeviceNativeOps().setElementThreshold(16384);
//        NativeOpsHolder.getInstance().getDeviceNativeOps().setTADThreshold(64);
//        MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(OUTPUT + "model-test01-1.txt");
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
                i++;
                if (i % 500 == 0) {
                    log.info("Now is at prefix: " + prefix + ", cursor:" + eDataSetIterator.cursor() + ", total :" + eDataSetIterator.totalExamples());
//                    executorService.submit(new HibernateRunner(i, model, eDataSetIteratorTest));
                }
            }
        });
        log.info("Starting training");

        for (int j = 0; j < nEpochs; j++) {
            multiLayerNetwork.fit(eDataSetIterator);
            eDataSetIterator.reset();
            executorService.submit(new HibernateRunner(j, multiLayerNetwork, eDataSetIteratorTest, prefix));
        }
        File file = new File("model-" + prefix + "-full" + ".txt");
//        File file = new File(OUTPUT + "model-" + prefix + "-full" + ".txt");
        file.createNewFile();
        ModelSerializer.writeModel(multiLayerNetwork, file, true);
    }

    private static void operationFunctionTestModel(Scanner scanner) {

    }

    public static void main(String[] args) throws IOException, Nd4jBackend.NoAvailableBackendException {

//        CudaEnvironment.getInstance().getConfiguration().setMaximumDeviceCache(5L * 1024 * 1024 * 1024);

        Scanner scanner = new Scanner(System.in);
//        System.out.println("Please select operation: 0:rebuild wordVector 1:rebuild model");
//        int operationCode = scanner.nextInt();
//        Preconditions.checkArgument(operationCode == 0 || operationCode == 1 || operationCode == 2);
//        if (operationCode == 0) {
//            operationFunctionRebuildWordVector(scanner);
//        } else if (operationCode == 1) {
//            operationFunctionRebuildModel(scanner);
//        } else {
//            operationFunctionTestModel(scanner);
//        }
        operationFunctionRebuildModel(scanner);
//        Runtime.getRuntime().exec("shutdown -s -t 10");
    }
}

class HibernateRunner implements Runnable {

    private final int anInt;
    private String path;
    private final String prefix;
    private final MultiLayerNetwork model;
    private final DataSetIterator dataSetIterator;

    public HibernateRunner(int anInt, Model model, DataSetIterator iterator, String prefix) {
        this.dataSetIterator = iterator;
        this.anInt = anInt;
        this.prefix = prefix;
//        this.path = EmojiLSTM.OUTPUT + "model-" + prefix + "-" + anInt + ".txt";
        this.path = "model-" + prefix + "-" + anInt + ".txt";
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
