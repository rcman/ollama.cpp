import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.io.*;
import java.util.*;

public class OllamaGUI extends JFrame {
    private final JTextArea outputArea = new JTextArea(20, 60);
    private final JTextField inputField = new JTextField(50);
    private final JButton sendButton = new JButton("Send");
    private final JComboBox<String> modelSelector = new JComboBox<>();
    private final JButton loadButton = new JButton("Load Model");
    private final JButton unloadButton = new JButton("Unload Model");
    private final JLabel statusLabel = new JLabel("Status: Not initialized");
    private final OllamaNativeInterface ollama = new OllamaNativeInterface();

    public OllamaGUI() {
        super("Ollama.cpp Java GUI");
        setupUI();
        setupEventHandlers();
        loadAvailableModels();
        initializeNative();
    }

    private void setupUI() {
        outputArea.setEditable(false);
        outputArea.setLineWrap(true);
        outputArea.setWrapStyleWord(true);
        
        JPanel inputPanel = new JPanel(new BorderLayout());
        inputPanel.add(inputField, BorderLayout.CENTER);
        inputPanel.add(sendButton, BorderLayout.EAST);
        
        JPanel modelPanel = new JPanel(new FlowLayout());
        modelPanel.add(new JLabel("Model:"));
        modelPanel.add(modelSelector);
        modelPanel.add(loadButton);
        modelPanel.add(unloadButton);
        
        JPanel controlPanel = new JPanel(new BorderLayout());
        controlPanel.add(modelPanel, BorderLayout.NORTH);
        controlPanel.add(statusLabel, BorderLayout.SOUTH);
        
        getContentPane().setLayout(new BorderLayout());
        getContentPane().add(new JScrollPane(outputArea), BorderLayout.CENTER);
        getContentPane().add(inputPanel, BorderLayout.SOUTH);
        getContentPane().add(controlPanel, BorderLayout.NORTH);
        
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
    }

    private void setupEventHandlers() {
        sendButton.addActionListener(e -> processInput());
        inputField.addActionListener(e -> processInput());
        
        loadButton.addActionListener(e -> {
            String model = (String) modelSelector.getSelectedItem();
            if (model != null) {
                loadModel(model);
            }
        });
        
        unloadButton.addActionListener(e -> unloadModel());
    }

    private void processInput() {
        String input = inputField.getText().trim();
        if (!input.isEmpty()) {
            outputArea.append("You: " + input + "\n");
            inputField.setText("");
            generateResponse(input);
        }
    }

    private void loadAvailableModels() {
        File modelDir = new File("models");
        if (modelDir.exists() && modelDir.isDirectory()) {
            String[] models = modelDir.list((dir, name) -> name.endsWith(".bin"));
            if (models != null) {
                for (String model : models) {
                    modelSelector.addItem(model);
                }
            }
        }
    }

    private void initializeNative() {
        if (ollama.initialize()) {
            updateStatus("Native library loaded successfully");
        } else {
            updateStatus("Failed to load native library");
        }
    }

    private void loadModel(String modelName) {
        if (ollama.loadModel("models/" + modelName)) {
            updateStatus("Model loaded: " + modelName);
        } else {
            updateStatus("Failed to load model: " + modelName);
        }
    }

    private void unloadModel() {
        ollama.unloadModel();
        updateStatus("Model unloaded");
    }

    private void generateResponse(String prompt) {
        new Thread(() -> {
            String response = ollama.generateText(prompt);
            SwingUtilities.invokeLater(() -> {
                outputArea.append("Ollama: " + response + "\n");
            });
        }).start();
    }

    private void updateStatus(String message) {
        statusLabel.setText("Status: " + message);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new OllamaGUI().setVisible(true));
    }
}

class OllamaNativeInterface {
    static {
        System.loadLibrary("ollama_jni");
    }

    public native boolean initialize();
    public native boolean loadModel(String modelPath);
    public native void unloadModel();
    public native String generateText(String prompt);
}