package me.peyppicp.cnn;

import me.peyppicp.advance2.WordToIndex;
import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class EmojiSampleProcess {

    public static void main(String[] args) throws IOException {
        File emojiSampleFile = new File("emoji_sample.txt");
        List<String> emojiSamples = FileUtils.readLines(emojiSampleFile, Charsets.UTF_8);
        WordToIndex wordToIndex = new WordToIndex("emoji_sample.txt");
        for (String sample : emojiSamples) {

        }
    }
}
