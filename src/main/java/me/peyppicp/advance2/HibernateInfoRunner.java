package me.peyppicp.advance2;

import org.apache.commons.io.Charsets;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;

/**
 * @author YuXiao Pan
 * @date 2017/12/12
 * @email yuxiao.pan@kikatech.com
 */
public class HibernateInfoRunner implements Runnable {
    private final int anInt;
    private String path;
    private String evaPath;
    private final String prefix;
    private final Model model;
    private final DataSetIterator dataSetIterator;
    private final Evaluation evaluation;

    public HibernateInfoRunner(int anInt, Model model, DataSetIterator iterator,
                               String prefix, Evaluation evaluation) {
        this.dataSetIterator = iterator;
        this.anInt = anInt;
        this.prefix = prefix;
        this.path = FullOperationMain.OUTPUT + "model-" + prefix + "-" + anInt + ".txt";
        this.evaPath = FullOperationMain.OUTPUT + "eva-" + prefix + "-" + anInt + ".txt";
//        this.path = "model-" + prefix + "-" + anInt + ".txt";
        this.model = model;
        this.evaluation = evaluation;
    }

    @Override
    public void run() {
        try {
//            Evaluation evaluate = model.evaluate(dataSetIterator);
//            System.out.println(evaluate.stats());

            File file = new File(path);
            if (file.exists()) {
                file.delete();
            }
            file.createNewFile();
            ModelSerializer.writeModel(model, file, true);
            File file1 = new File(evaPath);
            if (file1.exists()) {
                file1.delete();
            }
            file1.createNewFile();
            FileUtils.write(file1, evaluation.stats(), Charsets.UTF_8, false);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
