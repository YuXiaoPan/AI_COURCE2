package me.peyppicp.advance;

import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class TestMain {

    public static void main(String[] args) throws IOException {

        File file = new File("EmojiSampleLabels.txt");
        List<String> strings = FileUtils.readLines(file, Charsets.UTF_8);
        List<String> collect = strings.parallelStream().map(s -> s.split(",")[0].split("-")[1]).distinct().collect(Collectors.toList());
        System.out.println();


    }
}
