import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.image.*;
import java.io.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.Base64;
import java.util.List;

public class Main {

    // image resizing dimensions
    static int imageWidth = 50;
    static int imageHeight = 30;

    // convoluted width and height, 3 colors and vertical/horizontal edge detection for each
    static int inputSize = (imageWidth - 2) * (imageHeight - 2) * 3 * 2;

    // neural network layers
    static int hidden1 = 128;
    static int hidden2 = 32;

    // buckets
    static String[] labels = {"null", "space", "space2", "up", "down", "left", "right"};
    static int outputSize = labels.length;

    // weights and biases
    static double[][] w1 = new double[hidden1][inputSize];
    static double[][] w2 = new double[hidden2][hidden1];
    static double[][] w3 = new double[outputSize][hidden2];
    static double[] b1 = new double[hidden1];
    static double[] b2 = new double[hidden2];
    static double[] b3 = new double[outputSize];

    // training specifications
    static double learningRate = 0.005;
    static double dropoutRate = 0.3;
    static int epochCount = 30;

    // either train and test a model or actually run one previously trained
    static boolean loadFromFile = true;
    static final String modelSavePath = "data/models/model_weights.dat";
    static final String modelPath = "data/models/v0.4.dat";

    static Map<String, Integer> labelIndex = new HashMap<>();

    static {
        for (int i = 0; i < labels.length; i++) labelIndex.put(labels[i], i);
    }

    public static void main(String[] args) throws Exception {
        Random rand = new Random(0);

        if (loadFromFile) {
            ModelIO.load(modelPath, w1, w2, w3, b1, b2, b3);
            System.out.println("Model loaded.");
        } else {
            initWeights(rand);
        }

        List<double[]> inputs = new ArrayList<>();
        List<Integer> targets = new ArrayList<>();
        List<String> filePaths = new ArrayList<>();

        File baseDir = new File("data/images");
        Set<String> seenHashes = new HashSet<>();
        MessageDigest md = MessageDigest.getInstance("MD5");

        for (String label : labels) {
            File dir = new File(baseDir, label);
            for (File file : Objects.requireNonNull(dir.listFiles())) {
                BufferedImage img = ImageIO.read(file);
                if (img == null) continue;
                BufferedImage resized = resizeImage(img, imageWidth, imageHeight);
                byte[] pixelBytes = ((DataBufferByte) resized.getRaster().getDataBuffer()).getData();
                String hash = Base64.getEncoder().encodeToString(md.digest(pixelBytes));
                if (seenHashes.contains(hash)) continue;
                seenHashes.add(hash);
                inputs.add(imageToArray(resized));
                targets.add(labelIndex.get(label));
                filePaths.add(file.getAbsolutePath());
            }
        }

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) indices.add(i);
        Collections.shuffle(indices, rand);

        int split = (int) (0.8 * inputs.size());
        List<double[]> trainInputs = new ArrayList<>();
        List<Integer> trainTargets = new ArrayList<>();
        List<String> trainPaths = new ArrayList<>();
        List<double[]> testInputs = new ArrayList<>();
        List<Integer> testTargets = new ArrayList<>();
        List<String> testPaths = new ArrayList<>();

        for (int i = 0; i < inputs.size(); i++) {
            int idx = indices.get(i);
            if (i < split) {
                trainInputs.add(inputs.get(idx));
                trainTargets.add(targets.get(idx));
                trainPaths.add(filePaths.get(idx));
            } else {
                testInputs.add(inputs.get(idx));
                testTargets.add(targets.get(idx));
                testPaths.add(filePaths.get(idx));
            }
        }

        if (!loadFromFile) {
            for (int epoch = 0; epoch < epochCount; epoch++) {
                double loss = 0;
                int correct = 0;
                List<Integer> trainIndices = new ArrayList<>();
                for (int i = 0; i < trainInputs.size(); i++) trainIndices.add(i);
                Collections.shuffle(trainIndices, rand);

                for (int i : trainIndices) {
                    double[] x = trainInputs.get(i);
                    int target = trainTargets.get(i);
                    double[] h1 = new double[hidden1];
                    double[] h2 = new double[hidden2];
                    boolean[] mask1 = new boolean[hidden1];
                    boolean[] mask2 = new boolean[hidden2];

                    double[] y = forward(x, h1, h2, mask1, mask2, true, rand);
                    double[] probs = softmax(y);
                    int predicted = argmax(probs);
                    if (predicted == target) correct++;

                    double[] grad = new double[outputSize];
                    for (int j = 0; j < outputSize; j++)
                        grad[j] = probs[j] - (j == target ? 1 : 0);

                    double[] dh2 = new double[hidden2];
                    for (int j = 0; j < outputSize; j++) {
                        for (int k = 0; k < hidden2; k++) {
                            dh2[k] += grad[j] * w3[j][k];
                            w3[j][k] -= learningRate * grad[j] * h2[k];
                        }
                        b3[j] -= learningRate * grad[j];
                    }

                    double[] dh1 = new double[hidden1];
                    for (int j = 0; j < hidden2; j++) {
                        if (!mask2[j] || h2[j] <= 0) continue;
                        for (int k = 0; k < hidden1; k++) {
                            dh1[k] += dh2[j] * w2[j][k];
                            w2[j][k] -= learningRate * dh2[j] * h1[k];
                        }
                        b2[j] -= learningRate * dh2[j];
                    }

                    for (int j = 0; j < hidden1; j++) {
                        if (!mask1[j] || h1[j] <= 0) continue;
                        for (int k = 0; k < inputSize; k++) {
                            w1[j][k] -= learningRate * dh1[j] * x[k];
                        }
                        b1[j] -= learningRate * dh1[j];
                    }

                    loss += -Math.log(probs[target] + 1e-10);
                }

                System.out.printf("Epoch %d | Loss: %.4f | Accuracy: %.2f%%\n", epoch, loss / trainInputs.size(), 100.0 * correct / trainInputs.size());
            }

            ModelIO.save(modelSavePath, w1, w2, w3, b1, b2, b3);
            System.out.println("Model saved.");
            System.out.println("\nTesting");
            int testCorrect = 0;
            for (int i = 0; i < testInputs.size(); i++) {
                double[] x = testInputs.get(i);
                int target = testTargets.get(i);
                double[] h1 = new double[hidden1];
                double[] h2 = new double[hidden2];
                double[] y = forward(x, h1, h2, null, null, false, rand);
                double[] probs = softmax(y);
                int predicted = argmax(probs);
                if (predicted == target) testCorrect++;
                else {
                    System.out.printf("Misclassified (test) image: %s\n", testPaths.get(i));
                    System.out.printf("Predicted: %s | Actual: %s\n", labels[predicted], labels[target]);
                }
            }
            System.out.printf("Test Accuracy: %.2f%%\n", 100.0 * testCorrect / testInputs.size());
        } else {
            Robot robot = new Robot();
            while(MouseInfo.getPointerInfo().getLocation().y > 200) {
                BufferedImage screenshot = robot.createScreenCapture(new Rectangle(745, 513, 103, 60));

                BufferedImage resized = resizeImage(screenshot, imageWidth, imageHeight);
                byte[] pixelBytes = ((DataBufferByte) resized.getRaster().getDataBuffer()).getData();

                //String hash = Base64.getEncoder().encodeToString(md.digest(pixelBytes));
                //if (seenHashes.contains(hash))

                double[] x = imageToArray(resized);
                double[] h1 = new double[hidden1];
                double[] h2 = new double[hidden2];
                double[] y = forward(x, h1, h2, null, null, false, rand);
                int predicted = argmax(softmax(y));
                int key = KeyEvent.VK_SPACE;
                if (predicted == 3) {
                    key = KeyEvent.VK_UP;
                } else if (predicted == 4) {
                    key = KeyEvent.VK_DOWN;
                } else if (predicted == 5) {
                    key = KeyEvent.VK_LEFT;
                } else if (predicted == 6) {
                    key = KeyEvent.VK_RIGHT;
                }
                if (predicted > 0) {
                    robot.keyPress(key);
                    Thread.sleep(10);
                    robot.keyRelease(key);
                    Thread.sleep(10);
                } else {
                    Thread.sleep(20);
                }
            }
        }
    }

    static double[] forward(double[] x, double[] h1, double[] h2, boolean[] mask1, boolean[] mask2, boolean training, Random rand) {
        for (int i = 0; i < hidden1; i++) {
            h1[i] = b1[i];
            for (int j = 0; j < inputSize; j++) h1[i] += w1[i][j] * x[j];
            h1[i] = Math.max(0, h1[i]);
            if (training) {
                mask1[i] = rand.nextDouble() > dropoutRate;
                if (!mask1[i]) h1[i] = 0;
            }
        }

        for (int i = 0; i < hidden2; i++) {
            h2[i] = b2[i];
            for (int j = 0; j < hidden1; j++) h2[i] += w2[i][j] * h1[j];
            h2[i] = Math.max(0, h2[i]);
            if (training) {
                mask2[i] = rand.nextDouble() > dropoutRate;
                if (!mask2[i]) h2[i] = 0;
            }
        }

        double[] out = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            out[i] = b3[i];
            for (int j = 0; j < hidden2; j++) out[i] += w3[i][j] * h2[j];
        }
        return out;
    }

    static void initWeights(Random rand) {
        for (double[] row : w1)
            for (int i = 0; i < row.length; i++)
                row[i] = rand.nextGaussian() * Math.sqrt(2.0 / inputSize);
        for (double[] row : w2)
            for (int i = 0; i < row.length; i++)
                row[i] = rand.nextGaussian() * Math.sqrt(2.0 / hidden1);
        for (double[] row : w3)
            for (int i = 0; i < row.length; i++)
                row[i] = rand.nextGaussian() * Math.sqrt(2.0 / hidden2);
    }

    static double[] softmax(double[] z) {
        double max = Arrays.stream(z).max().orElse(0);
        double sum = 0;
        double[] result = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            result[i] = Math.exp(z[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < z.length; i++) result[i] /= sum;
        return result;
    }

    static int argmax(double[] array) {
        int maxIdx = 0;
        for (int i = 1; i < array.length; i++)
            if (array[i] > array[maxIdx]) maxIdx = i;
        return maxIdx;
    }

    /* //for just neural network w/o convolution
    static double[] imageToArray(BufferedImage img) {
        double[] arr = new double[inputSize];
        int idx = 0;
        for (int y = 0; y < img.getHeight(); y++) {
            for (int x = 0; x < img.getWidth(); x++) {
                int rgb = img.getRGB(x, y);
                arr[idx++] = ((rgb >> 16) & 0xFF) / 255.0;
                arr[idx++] = ((rgb >> 8) & 0xFF) / 255.0;
                arr[idx++] = (rgb & 0xFF) / 255.0;
            }
        }
        return arr;
    }*/

    // convert image to array with convolution and flattening
    static double[] imageToArray(BufferedImage img) {
        //RescaleOp rescaleOp = new RescaleOp(1.2f, 15, null);
        //rescaleOp.filter(img, img);
        int width = img.getWidth(), height = img.getHeight();
        double[][] red = new double[height][width];
        double[][] green = new double[height][width];
        double[][] blue = new double[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = img.getRGB(x, y);
                red[y][x] = ((rgb >> 16) & 0xFF) / 255.0;
                green[y][x] = ((rgb >> 8) & 0xFF) / 255.0;
                blue[y][x] = (rgb & 0xFF) / 255.0;
            }
        }

        double[][] kernel = {
                {-1, 0, 1},
                {-1, 0, 1},
                {-1, 0, 1}
        };

        double[][] kernel2 = {
                {-1, -1, -1},
                {0, 0, 0},
                {1, 1, 1}
        };

        // vertical edge detection
        double[][] convRed = convolve(red, kernel);
        double[][] convGreen = convolve(green, kernel);
        double[][] convBlue = convolve(blue, kernel);

        int outHeight = convRed.length;
        int outWidth = convRed[0].length;

        double[] flat = new double[outHeight * outWidth * 3 * 2];
        int idx = 0;
        for (int y = 0; y < outHeight; y++) {
            for (int x = 0; x < outWidth; x++) {
                flat[idx++] = Math.max(0, convRed[y][x]);
                flat[idx++] = Math.max(0, convGreen[y][x]);
                flat[idx++] = Math.max(0, convBlue[y][x]);
            }
        }

        // horizontal edge detection
        convRed = convolve(red, kernel2);
        convGreen = convolve(green, kernel2);
        convBlue = convolve(blue, kernel2);

        for (int y = 0; y < outHeight; y++) {
            for (int x = 0; x < outWidth; x++) {
                flat[idx++] = Math.max(0, convRed[y][x]);
                flat[idx++] = Math.max(0, convGreen[y][x]);
                flat[idx++] = Math.max(0, convBlue[y][x]);
            }
        }
        return flat;
    }

    // no padding since it lowers accuracy
    static double[][] convolve(double[][] input, double[][] kernel) {
        int kh = kernel.length, kw = kernel[0].length;
        int ih = input.length, iw = input[0].length;
        int oh = ih - kh + 1, ow = iw - kw + 1;
        double[][] output = new double[oh][ow];

        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                double sum = 0;
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        sum += kernel[ky][kx] * input[y + ky][x + kx];
                    }
                }
                output[y][x] = sum;
            }
        }
        return output;
    }

    static BufferedImage resizeImage(BufferedImage original, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        resized.getGraphics().drawImage(original, 0, 0, width, height, null);
        return resized;
    }
}

class ModelIO {
    static void save(String path, double[][] w1, double[][] w2, double[][] w3,
                     double[] b1, double[] b2, double[] b3) throws IOException {
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(path))) {
            write2D(out, w1);
            write2D(out, w2);
            write2D(out, w3);
            write1D(out, b1);
            write1D(out, b2);
            write1D(out, b3);
        }
    }

    static void load(String path, double[][] w1, double[][] w2, double[][] w3,
                     double[] b1, double[] b2, double[] b3) throws IOException {
        try (DataInputStream in = new DataInputStream(new FileInputStream(path))) {
            read2D(in, w1);
            read2D(in, w2);
            read2D(in, w3);
            read1D(in, b1);
            read1D(in, b2);
            read1D(in, b3);
        }
    }

    private static void write2D(DataOutputStream out, double[][] array) throws IOException {
        for (double[] row : array) for (double val : row) out.writeDouble(val);
    }

    private static void write1D(DataOutputStream out, double[] array) throws IOException {
        for (double val : array) out.writeDouble(val);
    }

    private static void read2D(DataInputStream in, double[][] array) throws IOException {
        for (int i = 0; i < array.length; i++)
            for (int j = 0; j < array[i].length; j++)
                array[i][j] = in.readDouble();
    }

    private static void read1D(DataInputStream in, double[] array) throws IOException {
        for (int i = 0; i < array.length; i++) array[i] = in.readDouble();
    }
}
