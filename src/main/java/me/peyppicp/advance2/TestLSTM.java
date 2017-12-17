package me.peyppicp.advance2;

import com.vdurmont.emoji.EmojiParser;
import me.peyppicp.Utils;

import java.io.IOException;
import java.util.List;

public class TestLSTM {

    public static void main(String[] args) throws IOException {

        List<String> strings = Utils.readLinesFromPath("more_standard_emoji_sample.txt");
        System.out.println(strings.parallelStream()
                .filter(s -> EmojiParser.extractEmojis(s).size() == 0).count());
    }
}
