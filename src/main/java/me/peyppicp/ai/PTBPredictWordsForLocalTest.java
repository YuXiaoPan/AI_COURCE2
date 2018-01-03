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

/**
 * @author YuXiao Pan
 * @date 2017/12/17
 * @email yuxiao.pan@kikatech.com
 */
public class PTBPredictWordsForLocalTest {

    public static final String UNKNOWN = "<unknown>";
    public static final String EMOJI = "<emoji>";
    public static final String END = "<end>";
    //        public static final String OUTPUT = "/home/peyppicp/output/";
//    public static final String PREFIX = "/home/peyppicp/data/new/";
//    public static final String PREFIX = "/home/panyuxiao/data/new/";
//    public static final String OUTPUT = "/home/panyuxiao/output/";
    public static final String PREFIX = "";
    public static final String OUTPUT = "";
    private static final int limitNum = 15000;
    private static final Logger log = LoggerFactory.getLogger(PTBPredictWordsForLocalTest.class);

    public static void main(String[] args) throws IOException {
        Nd4j.getMemoryManager().setAutoGcWindow(5000);
        File originData = new File(PREFIX + "test.txt");
//        if (!originData.exists()) {
//            preMain();
//        }

        String prefix = "peyppicp";
        int batchSize = 4;
        int nEpochs = 10;
        int numberSteps = 5;
        List<String> samples = Utils.readLinesFromPath(originData.getCanonicalPath());
        WordLimiter wordLimiter = new WordLimiter(samples, limitNum);
        wordLimiter.toFile(PREFIX + Constants.PAIR);
        WordToIndex wordToIndex = new WordToIndex(PREFIX + "pairTest.txt");
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(PREFIX + Constants._50D);
        PTBDataSetIterator rDataSetIterator = new PTBDataSetIterator(batchSize,
                numberSteps, samples, wordToIndex, word2Vec);
//        PTBDataSetIterator tDataSetIterator = new PTBDataSetIterator(false, truncateLength, batchSize,
//                numberSteps, Utils.readLinesFromPath("testForTest.txt"), wordToIndex, word2Vec);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.SINGLE)
                .inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                .seed(new Random().nextInt(4))
                .updater(Updater.ADAM)
                .regularization(true)
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
//                .learningRateScoreBasedDecayRate(0.01)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(rDataSetIterator.inputColumns()).nOut(50)
                        .activation(Activation.TANH).build())
                .layer(1, new GravesLSTM.Builder().nIn(50).nOut(100)
                        .activation(Activation.TANH).dropOut(0.5d).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(100).nOut(rDataSetIterator.totalOutcomes()).build())
                .pretrain(false)
                .backprop(true)
                .build();

//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();
//        uiServer.attach(statsStorage);

        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(conf);
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
//                if (i % 10 == 0) {
                double loss = model.score();
                double exp = Math.exp(loss / i);
                System.out.println("PPL:" + exp + ",iterator:" + i);
//                }
            }
        });

        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
//        RnnTest rnnTest = new RnnTest(word2Vec, wordToIndex, tokenizerFactory, multiLayerNetwork, null, 3);

        System.out.println("begin train");
        for (int j = 0; j < nEpochs; j++) {
            multiLayerNetwork.fit(rDataSetIterator);
            rDataSetIterator.reset();
//            rnnTest.generateTokensFromStr("happy birthday to you", 50);
        }
    }

    private static void preMain() throws IOException {
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
            List<String> extractEmojis = EmojiParser.extractEmojis(line);
            for (String token : tokens) {
                if (word2Vec.hasWord(token)) {
                    filteredTokens.add(token);
                } else {
//                    filteredTokens.add(UNKNOWN);
                }
            }
            if (!extractEmojis.isEmpty()) {
                filteredTokens.add(EMOJI);
            } else {
                filteredTokens.add(END);
            }
            StringBuilder stringBuilder = new StringBuilder();
            filteredTokens.forEach(s -> stringBuilder.append(s).append(" "));
            tempResults.add(stringBuilder.toString().trim());
        }

        WordLimiter wordLimiter = new WordLimiter(tempResults, limitNum);
        wordLimiter.toFile(PREFIX + Constants.PAIR);
//        List<String> finalResults = new ArrayList<>(tempResults.size());
        StringBuilder stringBuilder = new StringBuilder();
        for (String tempResult : tempResults) {
            List<String> tokens = tokenizerFactory.create(tempResult).getTokens();
            List<String> indexes = new ArrayList<>(tokens.size());
            for (String token : tokens) {
                if (wordLimiter.getWordIndex(token) != -1) {
                    indexes.add(token);
                } else {
//                    indexes.add(UNKNOWN);
                }
            }
            indexes.forEach(s -> stringBuilder.append(s).append(" "));
//            finalResults.add(stringBuilder.toString().trim());
        }
        List<String> data = Lists.newArrayList();
        data.add(stringBuilder.toString().trim());
        String output = PREFIX + Constants.MORE_STANDARD;
        Utils.writeLineToPath(data, output);
    }
}
