import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.border.*;

public class OllamaAdvancedGUI extends JFrame {
    // GUI Components
    private final JTextArea outputArea = new JTextArea(20, 70);
    private final JTextField inputField = new JTextField(60);
    private final JButton sendButton = new JButton("Generate");
    private final JButton clearButton = new JButton("Clear");
    private final JLabel statusLabel = new JLabel("Status: Ready");
    
    // Parameter Controls
    private final JSlider contextSlider = createSlider(512, 8192, 2048);
    private final JSlider tempSlider = createSlider(0, 100, 30);
    private final JSlider topKSlider = createSlider(0, 100, 40);
    private final JSlider topPSlider = createSlider(0, 100, 95);
    private final JSlider repeatSlider = createSlider(100, 200, 130);
    private final JSlider batchSlider = createSlider(1, 512, 512);
    
    // Model Info
    private final JLabel modelLabel = new JLabel("Current Model: None");
    
    // Native Interface
    private final OllamaNativeInterface ollama = new OllamaNativeInterface();

    public OllamaAdvancedGUI() {
        super("Ollama.cpp Advanced Controller");
        setupUI();
        setupEventHandlers();
        initializeNative();
    }

    private JSlider createSlider(int min, int max, int value) {
        JSlider slider = new JSlider(min, max, value);
        slider.setPaintTicks(true);
        slider.setPaintLabels(true);
        slider.setMajorTickSpacing((max - min) / 4);
        return slider;
    }

    private void setupUI() {
        // Configure main text area
        outputArea.setEditable(false);
        outputArea.setWrapStyleWord(true);
        outputArea.setLineWrap(true);
        outputArea.setFont(new Font("Monospaced", Font.PLAIN, 14));
        
        // Create parameter panel
        JPanel paramPanel = new JPanel(new GridLayout(0, 2, 5, 5));
        paramPanel.setBorder(new TitledBorder("Generation Parameters"));
        
        paramPanel.add(createLabeledControl("Context Size:", contextSlider));
        paramPanel.add(createLabeledControl("Temperature (x100):", tempSlider));
        paramPanel.add(createLabeledControl("Top-K:", topKSlider));
        paramPanel.add(createLabeledControl("Top-P (x100):", topPSlider));
        paramPanel.add(createLabeledControl("Repeat Penalty (x100):", repeatSlider));
        paramPanel.add(createLabeledControl("Batch Size:", batchSlider));
        
        // Create control panel
        JPanel controlPanel = new JPanel(new BorderLayout(5, 5));
        controlPanel.add(modelLabel, BorderLayout.NORTH);
        controlPanel.add(paramPanel, BorderLayout.CENTER);
        
        // Create input panel
        JPanel inputPanel = new JPanel(new BorderLayout(5, 5));
        inputPanel.add(inputField, BorderLayout.CENTER);
        
        JPanel buttonPanel = new JPanel(new GridLayout(1, 2, 5, 5));
        buttonPanel.add(clearButton);
        buttonPanel.add(sendButton);
        inputPanel.add(buttonPanel, BorderLayout.EAST);
        
        // Create status panel
        JPanel statusPanel = new JPanel(new BorderLayout());
        statusPanel.add(statusLabel, BorderLayout.CENTER);
        
        // Main layout
        Container contentPane = getContentPane();
        contentPane.setLayout(new BorderLayout(10, 10));
        contentPane.add(controlPanel, BorderLayout.NORTH);
        contentPane.add(new JScrollPane(outputArea), BorderLayout.CENTER);
        contentPane.add(inputPanel, BorderLayout.SOUTH);
        contentPane.add(statusPanel, BorderLayout.SOUTH);
        
        // Window configuration
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
        setMinimumSize(new Dimension(800, 600));
    }
    
    private JPanel createLabeledControl(String label, JSlider slider) {
        JPanel panel = new JPanel(new BorderLayout(5, 5));
        panel.add(new JLabel(label), BorderLayout.WEST);
        
        JLabel valueLabel = new JLabel(String.valueOf(slider.getValue()));
        valueLabel.setPreferredSize(new Dimension(50, 20));
        
        slider.addChangeListener(e -> {
            valueLabel.setText(String.valueOf(slider.getValue()));
        });
        
        JPanel valuePanel = new JPanel(new BorderLayout());
        valuePanel.add(slider, BorderLayout.CENTER);
        valuePanel.add(valueLabel, BorderLayout.EAST);
        
        panel.add(valuePanel, BorderLayout.CENTER);
        return panel;
    }

    private void setupEventHandlers() {
        sendButton.addActionListener(e -> generateResponse());
        
        clearButton.addActionListener(e -> outputArea.setText(""));
        
        inputField.addActionListener(e -> generateResponse());
        
        // Add real-time parameter change handlers
        ChangeListener paramChangeListener = e -> updateGenerationParams();
        contextSlider.addChangeListener(paramChangeListener);
        tempSlider.addChangeListener(paramChangeListener);
        topKSlider.addChangeListener(paramChangeListener);
        topPSlider.addChangeListener(paramChangeListener);
        repeatSlider.addChangeListener(paramChangeListener);
        batchSlider.addChangeListener(paramChangeListener);
    }

    private void initializeNative() {
        if (ollama.initialize()) {
            updateStatus("Native library initialized");
            loadDefaultModel();
        } else {
            updateStatus("Failed to load native library");
        }
    }
    
    private void loadDefaultModel() {
        if (ollama.loadModel("models/llama-2-7b.bin")) {
            modelLabel.setText("Current Model: llama-2-7b.bin");
            updateGenerationParams();
            updateStatus("Default model loaded");
        } else {
            updateStatus("Failed to load default model");
        }
    }
    
    private void updateGenerationParams() {
        ollama.setParams(
            contextSlider.getValue(),
            tempSlider.getValue() / 100.0f,
            topKSlider.getValue(),
            topPSlider.getValue() / 100.0f,
            repeatSlider.getValue() / 100.0f,
            batchSlider.getValue()
        );
    }

    private void generateResponse() {
        String prompt = inputField.getText().trim();
        if (!prompt.isEmpty()) {
            inputField.setText("");
            outputArea.append("> " + prompt + "\n");
            
            new Thread(() -> {
                String response = ollama.generateText(prompt);
                SwingUtilities.invokeLater(() -> {
                    outputArea.append(response + "\n\n");
                    outputArea.setCaretPosition(outputArea.getDocument().getLength());
                });
            }).start();
        }
    }

    private void updateStatus(String message) {
        statusLabel.setText("Status: " + message);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            OllamaAdvancedGUI gui = new OllamaAdvancedGUI();
            gui.setVisible(true);
        });
    }
}

class OllamaNativeInterface {
    static {
        System.loadLibrary("ollama_jni");
    }

    public native boolean initialize();
    public native boolean loadModel(String modelPath);
    public native void setParams(int ctx, float temp, int topK, float topP, float repeatPenalty, int batchSize);
    public native String generateText(String prompt);
}