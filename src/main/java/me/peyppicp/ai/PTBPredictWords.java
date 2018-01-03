package me.peyppicp.ai;

import com.google.common.collect.Lists;
import com.vdurmont.emoji.EmojiParser;
import me.peyppicp.Utils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * @author YuXiao Pan
 * @date 2017/12/17
 * @email yuxiao.pan@kikatech.com
 */
public class PTBPredictWords {

    public static final String UNKNOWN = "<unknown>";
    public static final String EMOJI = "<emoji>";
    public static final String END = "<end>";

    //    public static final String OUTPUT = "/home/peyppicp/output/";
//    public static final String PREFIX = "/home/peyppicp/data/new/";
//    public static String PREFIX = "/home/panyuxiao/data/new/";
//    public static String OUTPUT = "/home/panyuxiao/output/";
    public static String PREFIX = "";
    public static String OUTPUT = "";
    private static final int limitNum = 30000;
    private static final Logger log = LoggerFactory.getLogger(PTBPredictWords.class);

    /**
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
//        CudaEnvironment.getInstance().getConfiguration()
//                .setMaximumDeviceCacheableLength(1024 * 1024 * 1024L)
//                .setMaximumDeviceCache(6L * 1024 * 1024 * 1024L)
//                .setMaximumHostCacheableLength(1024 * 1024 * 1024L)
//                .setMaximumHostCache(6L * 1024 * 1024 * 1024L);

        PREFIX = args[0];
        OUTPUT = args[1];

        System.out.println(2);
        Nd4j.getMemoryManager().setAutoGcWindow(1000);

        File originData = new File(PREFIX + Constants.MORE_STANDARD);
        File pairFile = new File(PREFIX + Constants.PAIR);
        if (!originData.exists() || !pairFile.exists()) {
            log.info("Begin prepare data for train.");
            prepareDataForTrain();
            log.info("Finish prepare data for train.");
        }

        String prefix = "with-emoji";
        int batchSize = 64;
        int nEpochs = 10;
        int numberSteps = 30;
        List<String> samples = Utils.readLinesFromPath(originData.getCanonicalPath());
        WordToIndex wordToIndex = new WordToIndex(PREFIX + Constants.PAIR);
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(PREFIX + Constants._50D);
        PTBDataSetIterator rDataSetIterator = new PTBDataSetIterator(batchSize,
                numberSteps, samples, wordToIndex, word2Vec);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.SINGLE)
                .inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                .seed(new Random().nextInt(12))
                .updater(Updater.ADAM)
                .regularization(true)
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
//                .learningRate(0.1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(rDataSetIterator.inputColumns()).nOut(80)
                        .activation(Activation.TANH).build())
                .layer(1, new GravesLSTM.Builder().nIn(80).nOut(50).activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(50).nOut(rDataSetIterator.totalOutcomes()).build())
                .pretrain(false)
                .backprop(true)
                .build();

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(conf);
//        MultiLayerNetwork multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork("ptb11.txt");
        multiLayerNetwork.init();
        multiLayerNetwork.setListeners(new IterationListener() {
            @Override
            public boolean invoked() {
                return false;
            }

            @Override
            public void invoke() {

            }

            @Override
            public void iterationDone(Model model, int i) {
                double loss = model.score();
                double exp = Math.exp(loss / i);
                System.out.println("PPL:" + exp + ",iterator:" + i);
            }
        });

        log.info("Begin Train");
        for (int j = 0; j < nEpochs; j++) {
            multiLayerNetwork.fit(rDataSetIterator);
            rDataSetIterator.reset();
            ModelSerializer.writeModel(multiLayerNetwork, new File(OUTPUT + prefix + j + ".txt"), true);
        }
    }

    private static void prepareDataForTrain() throws IOException {
        String emojiSamples = PREFIX + Constants.ORIGINAL_DATA;
        String input = PREFIX + Constants.TEMP_DATA;
        Utils.processOriginalSamples(emojiSamples, input, true);
        List<String> lines = Utils.readLinesFromPath(input);
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(PREFIX + Constants._50D);
        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        List<String> tempResults = new ArrayList<>();
        for (String line : lines) {
            List<String> tokens = tokenizerFactory.create(line).getTokens();
            List<String> filteredTokens = new ArrayList<>(tokens.size());
            List<String> extractEmojis = EmojiParser.extractEmojis(line).stream().distinct().collect(Collectors.toList());
            for (String token : tokens) {
                if (word2Vec.hasWord(token)) {
                    filteredTokens.add(token);
                } else {
                    filteredTokens.add(UNKNOWN);
                }
            }
            if (!extractEmojis.isEmpty()) {
//                filteredTokens.add(EMOJI);
                filteredTokens.addAll(extractEmojis);
            } else {
//                filteredTokens.add(END);
            }
            StringBuilder stringBuilder = new StringBuilder();
            filteredTokens.forEach(s -> stringBuilder.append(s).append(" "));
            tempResults.add(stringBuilder.toString().trim());
        }

        WordLimiter wordLimiter = new WordLimiter(tempResults, limitNum);
        wordLimiter.toFile(PREFIX + Constants.PAIR);
        StringBuilder stringBuilder = new StringBuilder();
        for (String tempResult : tempResults) {
            List<String> tokens = tokenizerFactory.create(tempResult).getTokens();
            List<String> indexes = new ArrayList<>(tokens.size());
            for (String token : tokens) {
                if (wordLimiter.getWordIndex(token) != -1) {
                    indexes.add(token);
                } else {
                    indexes.add(UNKNOWN);
                }
            }
            indexes.forEach(s -> stringBuilder.append(s).append(" "));
        }
        List<String> data = Lists.newArrayList();
        data.add(stringBuilder.toString().trim());
        String output = PREFIX + Constants.MORE_STANDARD;
        Utils.writeLineToPath(data, output);
    }
}
