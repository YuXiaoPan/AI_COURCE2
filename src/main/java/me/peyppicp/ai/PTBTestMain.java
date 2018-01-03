package me.peyppicp.ai;

import me.peyppicp.Utils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author YuXiao Pan
 * @date 2017/12/28
 * @email yuxiao.pan@kikatech.com
 */
public class PTBTestMain {

    public static void main(String[] args) throws IOException {
        String prefix = "";
        String output = "";
        String ptbModelPath = "";
        String ptbWord2VecPath = "";
        String ptbPairPath = "";
        String cnnModelPath = "";
        String cnnWord2VecPath = "";
        String testDataPath = "";
        MultiLayerNetwork ptbModel = ModelSerializer.restoreMultiLayerNetwork(ptbModelPath);
        WordVectors ptbWordVectors = WordVectorSerializer.readWord2VecModel(ptbWord2VecPath);
        WordToIndex wordToIndex = new WordToIndex(ptbPairPath);

        ComputationGraph cnnModel = ModelSerializer.restoreComputationGraph(cnnModelPath);
        Word2Vec cnnWord2Vec = WordVectorSerializer.readWord2VecModel(cnnWord2VecPath);
        EmojiToIndex emojiToIndex = new EmojiToIndex();

        DefaultTokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        RnnTest rnnTest = new RnnTest(ptbWordVectors, cnnWord2Vec,
                wordToIndex, emojiToIndex, tokenizerFactory, ptbModel, cnnModel, 3);
        List<String> collect = Utils.readLinesFromPath(testDataPath)
                .stream().limit(5000).collect(Collectors.toList());
        for (String s : collect) {
            rnnTest.generateTokensFromStr(s, 200);
        }
//        int threadNum = 8;
//        List<List<String>> testDatas = new ArrayList<>();
//        for (int i = 0; i < threadNum; i++) {
//            testDatas.add(new ArrayList<>());
//        }
//        for (int i = 0; i < collect.size(); i++) {
//            int number = i % threadNum;
//            List<String> strings = testDatas.get(number);
//            strings.add(collect.get(i));
//        }
//
//        ExecutorService executorService = Executors.newFixedThreadPool(threadNum);
//        for (int i = 0; i < threadNum; i++) {
//            executorService.submit(new EvaluationRunner(ModelSerializer.restoreMultiLayerNetwork(prefix + "ptb2.txt"),
//                    wordVectors, new WordToIndex(prefix + "pair.txt"), testDatas.get(i), tokenizerFactory));
//        }
//
//        System.out.println("--------------");
//        System.out.println(PTBEvaluation.getInstance().getCorrectTop1Rate());
//        System.out.println(PTBEvaluation.getInstance().getCorrectTop3Rate());
    }
}

class EvaluationRunner implements Runnable {

    private MultiLayerNetwork multiLayerNetwork;
    private WordVectors wordVectors;
    private WordToIndex wordToIndex;
    private List<String> testData;
    private DefaultTokenizerFactory defaultTokenizerFactory;

    public EvaluationRunner(MultiLayerNetwork multiLayerNetwork, WordVectors wordVectors, WordToIndex wordToIndex, List<String> testData, DefaultTokenizerFactory defaultTokenizerFactory) {
        this.multiLayerNetwork = multiLayerNetwork;
        this.wordVectors = wordVectors;
        this.wordToIndex = wordToIndex;
        this.testData = testData;
        this.defaultTokenizerFactory = defaultTokenizerFactory;
    }

    @Override
    public void run() {
//        RnnTest rnnTest = new RnnTest(wordVectors, wordToIndex, defaultTokenizerFactory, multiLayerNetwork, null, 3);
//        for (String s : testData) {
//            rnnTest.generateTokensFromStr(s, 10);
//        }
    }
}
