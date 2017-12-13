package me.peyppicp.advance2;

import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author YuXiao Pan
 * @date 2017/12/12
 * @email yuxiao.pan@kikatech.com
 */
public class ReEnforcementEmojiSample {

    public static void main(String[] args) throws IOException {
        File file = new File("EmojiSample.txt");
        List<String> emojiSamples = FileUtils.readLines(file, Charsets.UTF_8);
        ArrayList<String> newData = new ArrayList<>();
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        for (String sample : emojiSamples) {
            List<String> tokens = tokenizerFactory.create(sample).getTokens();
            for (int i = 0; i < tokens.size() - 2; i++) {
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j <= i; j++) {
                    sb.append(tokens.get(j)).append(" ");
                }
                newData.add(sb.toString().trim());
            }
            newData.add(sample);
        }
        FileUtils.writeLines(new File("ReEnforcementEmojiSample.txt"),
                "UTF-8",
                newData,
                "\n",
                false);
    }
}
