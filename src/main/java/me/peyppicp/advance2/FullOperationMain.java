package me.peyppicp.advance2;

import com.google.common.base.Preconditions;
import com.vdurmont.emoji.Emoji;
import com.vdurmont.emoji.EmojiManager;
import com.vdurmont.emoji.EmojiParser;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * @author YuXiao Pan
 * @date 2017/12/12
 * @email yuxiao.pan@kikatech.com
 */
public class FullOperationMain {

    public static final String OUTPUT = "/home/peyppicp/output/";
    public static final String PREFIX = "/home/peyppicp/data/new/";
    //    public static final String PREFIX = "/home/panyuxiao/data/new/";
//    public static final String OUTPUT = "/home/panyuxiao/output/";
//    public static final String PREFIX = "";
//    public static final String OUTPUT = "";
    private static final Logger log = LoggerFactory.getLogger(FullOperationMain.class);
    private static final int truncateReviewsToLength = 20;
    private static final ExecutorService executorService = Executors.newFixedThreadPool(2);

    public static void main(String[] args) throws IOException, Nd4jBackend.NoAvailableBackendException {
        File file = new File(PREFIX + "emoji_sample.txt");
        String prefix = "full01";
        Scanner scanner = new Scanner(System.in);
        String path = "";
        if (!file.exists()) {
            System.out.println("Please specify the emoji_sample.txt's path");
            path = scanner.nextLine();
            File tempFile = new File(path);
            if (tempFile.exists()) {
                file = tempFile;
            } else {
                System.out.println("Error path. System exit.");
                System.exit(0);
            }
        }

        scanner.close();
        File emojiSampleFile = new File(PREFIX + "EmojiSample.txt");
        File emojiSampleLabelFile = new File(PREFIX + "EmojiSampleLabels.txt");
        File emijiSampleWithoutEmojiFile = new File(PREFIX + "EmojiSampleWithoutEmoji.txt");
        File lookUpTableFile = new File(PREFIX + "glove.twitter.27B.25d.txt");
        if (!(emojiSampleFile.exists() && emojiSampleLabelFile.exists()
                && emijiSampleWithoutEmojiFile.exists() && lookUpTableFile.exists())) {
            System.out.println("Begin process original samples.");
            processOriginalSamples(file);
//            processWord2Vec();
//            System.out.println("Begin enforcement emoji samples.");
//            enforcementEmojiSamples();
            System.out.println("Begin mark labels.");
            markLabels();
            System.out.println("Begin remove emojis.");
            removeEmojis();
        }
        train(emojiSampleFile, emojiSampleLabelFile, emijiSampleWithoutEmojiFile,
                lookUpTableFile, file, prefix);
    }

    private static void train(File emojiSampleFile, File emojiSampleLabelFile,
                              File emijiSampleWithoutEmojiFile, File lookUpTableFile,
                              File file, String prefix) throws IOException {

        int batchSize = 256;
        int nEpochs = 10;
        WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(lookUpTableFile);
        WordToIndex wordToIndex = new WordToIndex(emojiSampleFile.getCanonicalPath());

        EDataSetIterator eDataSetIterator = new EDataSetIterator(wordToIndex, emijiSampleWithoutEmojiFile.getCanonicalPath(),
                emojiSampleLabelFile.getCanonicalPath(), wordVectors,
                batchSize, truncateReviewsToLength, false);
        EDataSetIterator eDataSetIteratorTest = new EDataSetIterator(wordToIndex, emijiSampleWithoutEmojiFile.getCanonicalPath(),
                emojiSampleLabelFile.getCanonicalPath(), wordVectors,
                batchSize, truncateReviewsToLength, true);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(new Random().nextInt())
                .updater(Updater.RMSPROP)
                .regularization(true)
                .l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.01)
//                .learningRateDecayPolicy(LearningRatePolicy.Inverse)
//                .lrPolicyDecayRate(0.001)
//                .lrPolicyPower(0.75)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(eDataSetIterator.inputColumns()).nOut(48)
                        .activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(48)
                        .nOut(eDataSetIterator.totalOutcomes()).build())
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
            Evaluation evaluate = multiLayerNetwork.evaluate(eDataSetIteratorTest);
            testResult(eDataSetIteratorTest, multiLayerNetwork);
            eDataSetIterator.reset();
            executorService.submit(new HibernateInfoRunner(j, multiLayerNetwork, eDataSetIteratorTest, prefix, evaluate));

        }
//        File file = new File("model-" + prefix + "-full" + ".txt");
        File outPutFile = new File(OUTPUT + "model-" + prefix + "-full" + ".txt");
        file.createNewFile();
        ModelSerializer.writeModel(multiLayerNetwork, outPutFile, true);
    }

    private static void testResult(EDataSetIterator eDataSetIterator, MultiLayerNetwork multiLayerNetwork) {
        for (String line : eDataSetIterator.getTotalLines()) {
            List<String> tokens = eDataSetIterator.getTokenizerFactory().create(line).getTokens();
            List<String> tokenFiltered = new ArrayList<>();
            for (String t : tokens) {
                if (eDataSetIterator.getWordVectors().hasWord(t)) {
                    tokenFiltered.add(t);
                }
            }
            int outputLength = Math.max(truncateReviewsToLength, tokenFiltered.size());
            INDArray features = Nd4j.create(1, eDataSetIterator.getVectorSize(), outputLength);
            for (int j = 0; j < tokens.size() && j < truncateReviewsToLength; j++) {
                String token = tokens.get(j);
                INDArray vectorMatrix = eDataSetIterator.getWordVectors().getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vectorMatrix);
            }

            INDArray output = multiLayerNetwork.output(features, false);
            int size = output.size(2);
            INDArray probabilitiesAtLastWord = output.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(size - 1));

            Number number = probabilitiesAtLastWord.maxNumber();
            long l = number.longValue();
//            List<String> unqiueLabelList = eDataSetIterator.getUnqiueLabelList();
//            System.out.println(probabilitiesAtLastWord.toString());
//            System.out.println("\n-------------------------------");
//            System.out.println("review: " + line);
//            System.out.println("Prefer:" + emojiStr);
//            System.out.println("label: " + probabilitiesAtLastWord.getDouble(1));
        }
    }

    private static void removeEmojis() throws IOException {
        File file = new File(PREFIX + "EmojiSample.txt");
//        File file = new File(PREFIX + "EmojiSample.txt");
        List<String> samples = FileUtils.readLines(file, Charsets.UTF_8);
        ArrayList<String> result = new ArrayList<>();
        for (String sample : samples) {
            String s = EmojiParser.removeAllEmojis(sample);
            result.add(s.trim());
        }

        Preconditions.checkArgument(samples.size() == result.size());
        FileUtils.writeLines(new File(PREFIX + "EmojiSampleWithoutEmoji.txt"),
                "UTF-8",
                result,
                "\n",
                false);
    }

    private static void markLabels() throws IOException {
        File file = new File(PREFIX + "EmojiSample.txt");
//        File file = new File(PREFIX + "EmojiSample.txt");
        List<String> samples = FileUtils.readLines(file, Charsets.UTF_8);
        WordToIndex wordToIndex = new WordToIndex(PREFIX + "EmojiSample.txt");
        ArrayList<String> labels = new ArrayList<>();
        for (String sample : samples) {
            List<String> emojis = EmojiParser.extractEmojis(sample)
                    .parallelStream().distinct().collect(Collectors.toList());
            StringBuilder sb = new StringBuilder();
            if (emojis.size() == 0) {
                int index = wordToIndex.getIndex(WordToIndex.STOP);
                labels.add(String.valueOf(index));
                continue;
            }
            for (String emoji : emojis) {
                int index = wordToIndex.getIndex(emoji);
                sb.append(index).append(",");
            }
            sb.deleteCharAt(sb.length() - 1);
            labels.add(sb.toString());
        }
        Preconditions.checkArgument(samples.size() == labels.size());
        FileUtils.writeLines(new File(PREFIX + "EmojiSampleLabels.txt"),
                "UTF-8",
                labels,
                "\n",
                false);
//        FileUtils.writeLines(new File(PREFIX + "EmojiSampleLabels.txt"),
//                "UTF-8",
//                labels,
//                "\n",
//                false);
    }

    private static void enforcementEmojiSamples() throws IOException {
        File file = new File(PREFIX + "EmojiSample.txt");
        List<String> emojiSamples = FileUtils.readLines(file, Charsets.UTF_8);
        ArrayList<String> newData = new ArrayList<>();
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        for (String sample : emojiSamples) {
            List<String> tokens = tokenizerFactory.create(sample).getTokens();
            if (tokens.size() >= truncateReviewsToLength) {
                continue;
            }
            for (int i = 0; i < tokens.size() - 2; i++) {
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j <= i; j++) {
                    sb.append(tokens.get(j)).append(" ");
                }
                newData.add(sb.toString().trim());
            }
            StringBuilder stringBuilder = new StringBuilder();
            tokens.forEach(s -> stringBuilder.append(s).append(" "));
            newData.add(stringBuilder.toString().trim());
        }

        FileUtils.writeLines(new File(PREFIX + "ReEnforcementEmojiSample.txt"),
                "UTF-8",
                newData,
                "\n",
                false);
    }

    private static void processOriginalSamples(File file) throws IOException {
        List<String> sampleLines = FileUtils.readLines(file, Charsets.UTF_8);
        List<String> emojiUnicodes = EmojiManager.getAll().parallelStream().map(Emoji::getUnicode).collect(Collectors.toList());
        List<String> temp = new ArrayList<>();
        List<String> errorLines = new ArrayList<>();
        int count = 0;
        int totalSize = sampleLines.size() / 1000;

//        sampleLines = sampleLines.parallelStream().map(EmojiParser::parseToUnicode).collect(Collectors.toList());

//        按照emoji进行切分
        for (String line : sampleLines) {
            try {
                int emojiLength = 2;
                int currentEmojiIndex = 0;
                List<String> containedEmojis = EmojiParser.extractEmojis(line).parallelStream().distinct().collect(Collectors.toList());
                for (String emoji : containedEmojis) {
                    boolean flag = false;
                    currentEmojiIndex = line.indexOf(emoji);
                    if (currentEmojiIndex != -1) {
                        for (int i = currentEmojiIndex; i < line.length() - 1; i += emojiLength) {
                            if (EmojiManager.isEmoji(line.substring(currentEmojiIndex, currentEmojiIndex + emojiLength))) {
                                currentEmojiIndex += emojiLength;
                                flag = true;
                            }
                        }
                    }
                    if (flag) {
                        temp.add(line.substring(0, currentEmojiIndex).trim().toLowerCase());
                        line = line.substring(currentEmojiIndex, line.length()).trim().toLowerCase();
                    }
                }
            } catch (Exception e) {
                errorLines.add(line);
            }
            count++;
            if (count % 1000 == 0) {
                System.out.println("Remain: " + (totalSize - (count / 1000)));
            }
        }

        temp = temp.parallelStream().filter(s -> EmojiParser.extractEmojis(s).size() != s.length() / 2).distinct().collect(Collectors.toList());
        List<String> temp1 = new ArrayList<>();

//        添加空格
        for (String sample : temp) {
            String emoji = EmojiParser.extractEmojis(sample).get(0);
            int i = sample.indexOf(emoji);
            if (i >= 1) {
                if (i == emoji.length() - 1) {
                    continue;
                } else {
                    String head = sample.substring(0, i);
                    String last = sample.substring(i, sample.length());
                    temp1.add(head + " " + last);
                }
            }
        }

        FileUtils.writeLines(new File(PREFIX + "EmojiSample.txt"),
                "UTF-8",
                temp1,
                "\n",
                false);
    }

    private void generateCNNData() {

    }

}
