using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEngine.SceneManagement;
#endif

public class VRMainMenuManager : MonoBehaviour
{
    public GameObject grabbableObject;
    public GameObject unloadingArea; 
    public GameObject grabbableObjectBlindArea;
    public GameObject windArea;
    public GameObject directionalLight; 
    public GameObject[] industrialLights; 

    public GameObject basicButton;
    public GameObject blindPickButton;
    public GameObject windButton;
    public GameObject lowLightButton;
    public GameObject powerLineButton;
    public GameObject trainingButton1; // Button for the first Training scenario
    public GameObject trainingButton2; // Button for the second Training scenario
    public GameObject trainingButton3; // Button for the third Training scenario
    public GameObject finishButton;

    public GameObject craneHook;

    private LowLightHandler lowLightHandler;
    public Metrics_Calculator metricsCalculator;
    public CSVWriter csvWriter;
    public VREyeTrackingDataManager vrEyeTracker; // VR eye tracking manager

    private void Start()
    {
        SetGameObjects(false, false, false);
        SetIndustrialLights(false);
        if (directionalLight != null)
            lowLightHandler = directionalLight.GetComponent<LowLightHandler>();
        if (lowLightHandler != null)
            lowLightHandler.enabled = false;

        SetMenuVisibility(true);
    }

    public void StartTrainingScenario1()
    {
        grabbableObject.transform.localPosition = new Vector3(-49.4700012f, 4.37500048f, 44.7400017f);
        unloadingArea.transform.localPosition = new Vector3(-90.9000015f, 47f, -41.7000008f);
        
        SetupTrainingScenario("Training1Scenario");
    }

    public void StartTrainingScenario2()
    {
        grabbableObject.transform.localPosition = new Vector3(-49.4699211f,2.37753153f,51.0600014f);
        unloadingArea.transform.localPosition = new Vector3(-90.9000015f,47f,11f);

        SetupTrainingScenario("Training2Scenario");
    }

    public void StartTrainingScenario3()
    {
        grabbableObject.transform.localPosition = new Vector3(-48.6800003f,4.37500143f,40.9900017f);
        unloadingArea.transform.localPosition = new Vector3(-93.4000015f,47f,36.2999992f);

        SetupTrainingScenario("Training3Scenario");
    }

    private void SetupTrainingScenario(string scenarioName)
    {
        // Set delay timer in DustCloudTrigger

        // Set up the game objects needed for the training scenario
        SetGameObjects(true, false, false);

        // Disable all objects in the public list
        SetObjectsActive(false);

        // Set the scenario flag to Blind Pick
        metricsCalculator.isBlindPickScenario = false;

        // Set the menu visibility and initialize the crane hook
        SetMenuVisibility(false);
        InitializeCraneHook();
        SetIndustrialLights(false);
        if (lowLightHandler != null)
            lowLightHandler.enabled = false;

        csvWriter.InitializeCSVWriter(scenarioName);
        metricsCalculator.InitializeMetricsCalculator(scenarioName);
        vrEyeTracker.InitializeVREyeTracking(scenarioName); // Start VR eye tracking data export
    }

    public void StartBasicScenario()
    {
        SetGameObjects(true, false, false);
        SetMenuVisibility(false);
        InitializeCraneHook();
        SetIndustrialLights(false);
        // Set delay timer in DustCloudTrigger
        if (lowLightHandler != null)
            lowLightHandler.enabled = false;
        // Set the scenario flag to Basic
        metricsCalculator.isBlindPickScenario = false;
        csvWriter.InitializeCSVWriter("BaselineScenario");
        metricsCalculator.InitializeMetricsCalculator("BaselineScenario");
        vrEyeTracker.InitializeVREyeTracking("BaselineScenario"); // Start VR eye tracking data export
    }

    public void StartBlindPickScenario()
    {
        SetGameObjects(false, true, false);
        SetMenuVisibility(false);
        InitializeCraneHookBlindPick();
        SetIndustrialLights(false);
        // Set delay timer in DustCloudTrigger
        if (lowLightHandler != null)
            lowLightHandler.enabled = false;
        // Set the scenario flag to Basic
        metricsCalculator.isBlindPickScenario = true;
        csvWriter.InitializeCSVWriter("BlindPickScenario");
        metricsCalculator.InitializeMetricsCalculator("BlindPickScenario");
        vrEyeTracker.InitializeVREyeTracking("BlindPickScenario"); // Start VR eye tracking data export
    }

    public void StartWindScenario()
    {
        SetGameObjects(true, false, true);
        SetMenuVisibility(false);
        InitializeCraneHook();
        SetIndustrialLights(false);
        // Set delay timer in DustCloudTrigger
        if (lowLightHandler != null)
            lowLightHandler.enabled = false;
        // Set the scenario flag to Basic
        metricsCalculator.isBlindPickScenario = false;
        csvWriter.InitializeCSVWriter("WindScenario");
        metricsCalculator.InitializeMetricsCalculator("WindScenario");
        vrEyeTracker.InitializeVREyeTracking("WindScenario"); // Start VR eye tracking data export
    }

    public void StartLowLightScenario()
    {
        SetGameObjects(true, false, false);
        SetMenuVisibility(false);
        InitializeCraneHook();
        SetIndustrialLights(true);
        // Set delay timer in DustCloudTrigger
        if (lowLightHandler != null)
            lowLightHandler.enabled = true;
        // Set the scenario flag to Basic
        metricsCalculator.isBlindPickScenario = false;
        csvWriter.InitializeCSVWriter("LowLightScenario");
        metricsCalculator.InitializeMetricsCalculator("LowLightScenario");
        vrEyeTracker.InitializeVREyeTracking("LowLightScenario"); // Start VR eye tracking data export
    }

    public void SwitchToPowerlineScene()
    {
        SceneManager.LoadScene("PowerLineScene"); // Load the PowerLineScene
    }
    
    private void SetGameObjects(bool grabbable, bool blindArea, bool wind)
    {
        grabbableObject.SetActive(grabbable);
        grabbableObjectBlindArea.SetActive(blindArea);
        if (windArea != null)
            windArea.GetComponent<WindArea>().enabled = wind;
    }

    private void SetMenuVisibility(bool visible, bool showFinishButton = true)
    {
        basicButton.SetActive(visible);
        blindPickButton.SetActive(visible);
        windButton.SetActive(visible);
        lowLightButton.SetActive(visible);
        powerLineButton.SetActive(visible);
        trainingButton1.SetActive(visible);
        trainingButton2.SetActive(visible); // Make the Training 2 button visible
        trainingButton3.SetActive(visible); // Make the Training 3 button visible
        finishButton.SetActive(showFinishButton);
    }

    public void FinishGame()
    {
        SetGameObjects(false, false, false);
        SetMenuVisibility(true, false); // Make only the finish button visible
        SetIndustrialLights(false);
        if (lowLightHandler != null)
            lowLightHandler.enabled = false;

        // Export metrics when the Finish button is clicked
        if (metricsCalculator != null)
        {
            metricsCalculator.ExportMetrics();
        }

#if UNITY_EDITOR
        EditorApplication.isPlaying = false;
#endif
    }
    
    private void InitializeCraneHook()
    {
        SignalPerson signalPerson = craneHook.GetComponent<SignalPerson>();
        if (signalPerson != null)
        {
            signalPerson.InitializeGrabbableObject();
        }
    }

    private void InitializeCraneHookBlindPick()
    {
        SignalPersonBlindPick signalPerson = craneHook.GetComponent<SignalPersonBlindPick>();
        if (signalPerson != null)
        {
            signalPerson.InitializeGrabbableObject();
        }
    }

    private void SetIndustrialLights(bool isActive)
    {
        foreach (GameObject light in industrialLights)
        {
            light.GetComponent<IndustrialLightManager>().SetLightsActive(isActive);
        }
    }

    // Method to disable objects in the training scenario
    private void SetObjectsActive(bool isActive)
    {
        // Find all objects with the "TrainingDisable" tag
        GameObject[] objectsToDisable = GameObject.FindGameObjectsWithTag("TrainingDisable");
        foreach (GameObject obj in objectsToDisable)
        {
            obj.SetActive(isActive);
        }
    }
}













// using UnityEngine;
// #if UNITY_EDITOR
// using UnityEditor;
// #endif

// public class MainMenuManager : MonoBehaviour
// {
//     public GameObject grabbableObject;
//     public GameObject grabbableObjectBlindArea;
//     public GameObject windArea;

//     public GameObject basicButton;
//     public GameObject blindPickButton;
//     public GameObject windButton;
//     public GameObject finishButton;

//     public GameObject craneHook;

//     private void Start()
//     {
//         // Initially disable all game objects
//         SetGameObjects(false, false, false);

//         // Ensure all buttons are visible
//         SetMenuVisibility(true);
//     }

//     public void StartBasicScenario()
//     {
//         SetGameObjects(true, false, false);
//         SetMenuVisibility(false);
//         InitializeCraneHook();
//     }

//     public void StartBlindPickScenario()
//     {
//         SetGameObjects(false, true, false);
//         SetMenuVisibility(false);
//         InitializeCraneHook();
//     }

//     public void StartWindScenario()
//     {
//         SetGameObjects(true, false, true);
//         SetMenuVisibility(false);
//         InitializeCraneHook();
//     }

//     public void FinishGame()
//     {
//         SetGameObjects(false, false, false);
//         SetMenuVisibility(true);
//         // Add this line to stop the play mode in the editor
// #if UNITY_EDITOR
//         EditorApplication.isPlaying = false;
// #endif
//     }

//     private void SetGameObjects(bool grabbable, bool blindArea, bool wind)
//     {
//         grabbableObject.SetActive(grabbable);
//         grabbableObjectBlindArea.SetActive(blindArea);
//         if (windArea != null)
//         {
//             windArea.GetComponent<WindArea>().enabled = wind;
//         }
//     }

//     private void SetMenuVisibility(bool visible)
//     {
//         basicButton.SetActive(visible);
//         blindPickButton.SetActive(visible);
//         windButton.SetActive(visible);
//     }

//     private void InitializeCraneHook()
//     {
//         SignalPerson signalPerson = craneHook.GetComponent<SignalPerson>();
//         if (signalPerson != null)
//         {
//             signalPerson.InitializeGrabbableObject();
//         }
//     }
// }