import java.io.File;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

class Layer {
	public int mapNum, mapSize, filterSize, scaleSize;
	public double[][][][] filters, maps, loss;
	public double[] bias;
	public Layer() {}
}

public class tinyCNN {
	public String trainFile = "/home/vietbt/java/mnist_digits_train.txt";
	public String testFile = "/home/vietbt/java/mnist_digits_test.txt";
	public double learningRate = 0.55;
	public int batchSize = 50;
	public int outputSize = 10;
	private double[][][] trainData, testData;
	private double[] trainLabel, testLabel;
	private Layer layer1 = new Layer(); // Input
	private Layer layer2 = new Layer(); // Conv
	private Layer layer3 = new Layer(); // Pool
	private Layer layer4 = new Layer(); // Conv
	private Layer layer5 = new Layer(); // Pool
	private Layer layer6 = new Layer(); // Output
	private int batchIndex = 0;
	private Random r = new Random();

	public static void main(String[] args) throws Exception {
		new tinyCNN().run();
	}

	public void run() throws Exception {
		readTrainFile(trainFile);
		readTestFile(testFile);
		setup();
		train();
	}

	public void train() throws Exception {
		for (int i = 0;; i++) {
			List<Integer> listIndex = IntStream.range(0, trainData.length).boxed().collect(Collectors.toList());
			Collections.shuffle(listIndex);
			for (int index : listIndex.subList(0, batchSize)) {
				trainData(trainData[index], trainLabel[index]);
				batchIndex++;
			}
			update();
			batchIndex = 0;
			if (i % 100 == 0) System.out.println("\nStep\t" + i + " \tTestAccuracy =\t" + testPredict() + "%\tLearningRate =\t" + learningRate);
			else System.out.print(".");
		}
	}

	public double testPredict() {
		int right = testData.length;
		for (int i = 0; i < testData.length; i++) {
			forward(testData[i]);
			for (int j = 0; j < layer6.mapNum; j++)
				if (layer6.maps[batchIndex][j][0][0] > layer6.maps[batchIndex][(int) testLabel[i]][0][0]) {
					right--;
					break;
				}
		}
		return 100.0 * right / testData.length;
	}

	public void readTrainFile(String filePath) throws Exception {
		List<String> lines = Files.readAllLines(new File(filePath).toPath());
		trainData = new double[lines.size()][28][28];
		trainLabel = new double[lines.size()];
		readFile(lines, trainData, trainLabel);

	}

	public void readTestFile(String filePath) throws Exception {
		List<String> lines = Files.readAllLines(new File(filePath).toPath());
		testData = new double[lines.size()][28][28];
		testLabel = new double[lines.size()];
		readFile(lines, testData, testLabel);
	}

	public void readFile(List<String> lines, double[][][] data, double[] label) {
		int index = 0;
		for (String line : lines) {
			double[] arr = Arrays.stream(line.split("\\|")).mapToDouble(Double::parseDouble).toArray();
			int k = 0;
			for (int i = 0; i < 28; i++) 
				for (int j = 0; j < 28; j++) 
					data[index][i][j] = arr[k++];
			label[index++] = arr[k];
		}
	}

	public void setup() {
		layer1.mapNum = 1;
		layer1.mapSize = 28;
		layer1.maps = new double[batchSize][layer1.mapNum][layer1.mapSize][layer1.mapSize];
		setupConv(layer2, layer1, 6, 5);
		setupPool(layer3, layer2, 2);
		setupConv(layer4, layer3, 12, 5);
		setupPool(layer5, layer4, 2);
		layer6.mapSize = 1;
		setupConv(layer6, layer5, outputSize, layer5.mapSize);
	}

	public void setupConv(Layer layer, Layer pLayer, int mapNum, int size) {
		layer.mapNum = mapNum;
		layer.filterSize = size;
		layer.mapSize = pLayer.mapSize - size + 1;
		layer.filters = new double[pLayer.mapNum][mapNum][size][size];
		for (int i = 0; i < pLayer.mapNum; i++) 
			for (int j = 0; j < layer.mapNum; j++) 
				layer.filters[i][j] = random(layer.filterSize, size);
		layer.bias = new double[layer.mapNum];
		layer.loss = new double[batchSize][layer.mapNum][layer.mapSize][layer.mapSize];
		layer.maps = new double[batchSize][layer.mapNum][layer.mapSize][layer.mapSize];
	}

	public void setupPool(Layer layer, Layer pLayer, int scale) {
		layer.scaleSize = scale;
		layer.mapNum = pLayer.mapNum;
		layer.mapSize = pLayer.mapSize / layer.scaleSize;
		layer.loss = new double[batchSize][layer.mapNum][layer.mapSize][layer.mapSize];
		layer.maps = new double[batchSize][layer.mapNum][layer.mapSize][layer.mapSize];
	}

	public void trainData(double[][] data, double label) {
		forward(data);
		backPropagation(label);
	}

	public void forward(double[][] data) {
		layer1.maps[batchIndex][0] = data;
		setConvOutput(layer2, layer1);
		setSampOutput(layer3, layer2);
		setConvOutput(layer4, layer3);
		setSampOutput(layer5, layer4);
		setConvOutput(layer6, layer5);
	}

	public void setConvOutput(Layer layer, Layer pLayer) {
		for (int j = 0; j < layer.mapNum; j++) {
			double[][] sum = null;
			for (int i = 0; i < pLayer.mapNum; i++) {
				double[][] convn = conv(pLayer.maps[batchIndex][i], layer.filters[i][j]);
				sum = (sum == null) ? convn : plus(convn, sum);
			}
			layer.maps[batchIndex][j] = sigmodBias(sum, layer.bias[j]);
		}
	}

	public void setSampOutput(Layer layer, Layer pLayer) {
		for (int i = 0; i < pLayer.mapNum; i++) {
			layer.maps[batchIndex][i] = scale(pLayer.maps[batchIndex][i], layer.scaleSize);
		}
	}

	public void backPropagation(double label) {
		setOutputLoss(label);
		setDataErrors();
	}

	public void setDataErrors() {
		setSampLoss(layer5, layer6);
		setConvLoss(layer4, layer5);
		setSampLoss(layer3, layer4);
		setConvLoss(layer2, layer3);
	}

	public void setSampLoss(Layer layer, Layer nLayer) {
		for (int i = 0; i < layer.mapNum; i++) {
			double[][] sum = null;
			for (int j = 0; j < nLayer.mapNum; j++) {
				double[][] convFull = fullyConv(nLayer.loss[batchIndex][j], rotate(nLayer.filters[i][j]));
				sum = (sum == null) ? convFull : plus(convFull, sum);
			}
			layer.loss[batchIndex][i] = sum;
		}
	}

	public void setConvLoss(Layer layer, Layer nLayer) {
		for (int m = 0; m < layer.mapNum; m++) {
			double[][] nError = nLayer.loss[batchIndex][m];
			double[][] outMatrix = mul2(layer.maps[batchIndex][m]);
			layer.loss[batchIndex][m] = mul(outMatrix, kronecker(nError, nLayer.scaleSize));
		}
	}

	public void setOutputLoss(double label) {
		double[] target = new double[layer6.mapNum];
		target[(int) label] = 1;
		for (int m = 0; m < layer6.mapNum; m++) {
			double out = layer6.maps[batchIndex][m][0][0];
			layer6.loss[batchIndex][m][0][0] = out * (1 - out) * (target[m] - out);
		}
	}

	public void update() {
		updateFilters(layer2, layer1);
		updateBias(layer2);
		updateFilters(layer4, layer3);
		updateBias(layer4);
		updateFilters(layer6, layer5);
		updateBias(layer6);
	}

	public void updateBias(Layer layer) {
		for (int i = 0; i < layer.mapNum; i++)
			layer.bias[i] += learningRate * sum(layer.loss, i) / batchSize;
	}

	public void updateFilters(Layer layer, Layer pLayer) {
		for (int i = 0; i < layer.mapNum; i++) 
			for (int j = 0; j < pLayer.mapNum; j++) {
				double[][] deltaF = null;
				for (int k = 0; k < batchSize; k++) {
					double[][] delta = conv(pLayer.maps[k][j], layer.loss[k][i]);
					deltaF = (deltaF == null) ? delta : plus(delta, deltaF);
				}
				layer.filters[j][i] = mul3(layer.filters[j][i], div(deltaF));
			}
	}

	public double[][] random(int x, int y) {
		double[][] result = new double[x][y];
		for (int i = 0; i < x; i++)
			for (int j = 0; j < y; j++) 
				result[i][j] = r.nextDouble() - 0.5;
		return result;
	}

	public double[][] sigmodBias(double[][] m, double bias) {
		for (double[] x : m)
			for (int j = 0; j < m[0].length; j++)
				x[j] = 1 / (1 + Math.pow(Math.E, -bias - x[j]));
		return m;

	}

	public double[][] plus(double[][] x, double[][] y) {
		for (int i = 0; i < x.length; i++)
			for (int j = 0; j < x[0].length; j++)
				y[i][j] += x[i][j];
		return y;
	}

	public double[][] scale(double[][] m, int s) {
		double[][] result = new double[m.length / s][m[0].length / s];
		for (int i = 0; i < m.length / s; i++) 
			for (int j = 0; j < m[0].length / s; j++) 
				for (int p = 0; p < s; p++) 
					for (int q = 0; q < s; q++) 
						result[i][j] += m[i * s + p][j * s + q] / s / s;
		return result;
	}

	public double[][] rotate(double[][] m) {
		int x = m.length;
		int y = m[0].length;
		double[][] a = new double[x][y];
		for (int i = 0; i < x; i++) System.arraycopy(m[i], 0, a[i], 0, y);
		for (int i = 0; i < x; i++) 
			for (int j = 0; j < y / 2; j++) 
				a[i][j] = (a[i][j] + a[i][y - 1 - j]) - (a[i][y - 1 - j] = a[i][j]);
		for (int j = 0; j < y; j++) 
			for (int i = 0; i < x / 2; i++) 
				a[i][j] = (a[i][j] + a[x - 1 - i][j]) - (a[x - 1 - i][j] = a[i][j]);
		return a;
	}

	public double[][] fullyConv(double[][] m, double[][] f) {
		double[][] result = new double[m.length + 2 * f.length - 2][m[0].length + 2 * f[0].length - 2];
		for (int i = 0; i < m.length; i++) 
			for (int j = 0; j < m[0].length; j++) 
				result[i + f.length - 1][j + f[0].length - 1] = m[i][j];
		return conv(result, f);
	}

	public double[][] mul(double[][] x, double[][] y) {
		for (int i = 0; i < x.length; i++)
			for (int j = 0; j < x[0].length; j++)
				y[i][j] *= x[i][j];
		return y;
	}

	public double[][] kronecker(double[][] m, int s) {
		double[][] result = new double[m.length * s][m[0].length * s];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++) 
				for (int p = 0; p < s; p++)
					for (int q = 0; q < s; q++)
						result[i * s + p][j * s + q] = m[i][j];
		return result;
	}

	public double[][] mul2(double[][] x) {
		double[][] y = new double[x.length][x[0].length];
		for (int i = 0; i < x.length; i++)
			for (int j = 0; j < x[0].length; j++)
				y[i][j] = x[i][j] * (1 - x[i][j]);
		return y;
	}

	public double[][] div(double[][] m) {
		for (double[] x : m)
			for (int j = 0; j < m[0].length; j++)
				x[j] = x[j] / batchSize;
		return m;

	}

	public double[][] mul3(double[][] x, double[][] y) {
		for (int i = 0; i < x.length; i++)
			for (int j = 0; j < x[0].length; j++)
				y[i][j] = x[i][j] + y[i][j] * learningRate;
		return y;
	}

	public double sum(double[][][][] arr, int x) {
		double sum = 0;
		for (int i = 0; i < arr[0][x].length; i++)
			for (int j = 0; j < arr[0][x][0].length; j++) 
				for (double[][][] s : arr) 
					sum += s[x][i][j];
		return sum;
	}

	public double[][] conv(double[][] m, double[][] f) {
		int x = m.length - f.length + 1;
		int y = m[0].length - f[0].length + 1;
		double[][] result = new double[x][y];
		for (int i = 0; i < x; i++)
			for (int j = 0; j < y; j++) {
				double sum = 0.0;
				for (int p = 0; p < f.length; p++)
					for (int q = 0; q < f[0].length; q++)
						sum += m[i + p][j + q] * f[p][q];
				result[i][j] = sum;
			}
		return result;
	}
}
