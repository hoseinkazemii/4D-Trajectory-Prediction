using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class VRMainMenuManagerPowerLine : MonoBehaviour
{
    public GameObject grabbableObject;
    public GameObject grabbableObjectBlindArea;
    public GameObject windArea;
    public GameObject directionalLight; 
    public GameObject[] industrialLights; 

    public GameObject powerLineButton;
    public GameObject finishButton;

    public GameObject craneHook;

    private LowLightHandler lowLightHandler;
    public Metrics_Calculator_PowerLine metricsCalculatorPowerLine;
    public CSVWriter csvWriter;
    public VREyeTrackingDataManagerPowerLine vrEyeTrackerPowerLine; // VR eye tracking manager

    private void Start()
    {
        SetGameObjects(false, false, false);
        SetIndustrialLights(false);
        if (directionalLight != null)
            lowLightHandler = directionalLight.GetComponent<LowLightHandler>();
        if (lowLightHandler != null)
            lowLightHandler.enabled = false;

        // Set all buttons as visible
        SetMenuVisibility(true);
    }

    public void StartPowerLineScenario()
    {
        SetGameObjects(true, false, false);
        SetMenuVisibility(false);
        InitializeCraneHook();
        SetIndustrialLights(false);
        if (lowLightHandler != null)
            lowLightHandler.enabled = false;

        csvWriter.InitializeCSVWriter("PowerLineScenario"); // Initialize CSVWriter with scenario name
        metricsCalculatorPowerLine.InitializeMetricsCalculator("PowerLineScenario"); // Initialize MetricsCalculator with scenario name
        vrEyeTrackerPowerLine.InitializeVREyeTracking("PowerLineScenario"); // Initialize PCEyeTracker with scenario name
    }

    public void FinishGame()
    {
        SetGameObjects(false, false, false);
        SetMenuVisibility(true, false); // Make only the finish button visible
        SetIndustrialLights(false);
        if (lowLightHandler != null)
            lowLightHandler.enabled = false;

        // Export metrics when the Finish button is clicked
        if (metricsCalculatorPowerLine != null)
        {
            metricsCalculatorPowerLine.ExportMetrics();
        }

#if UNITY_EDITOR
        EditorApplication.isPlaying = false;
#endif
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
        powerLineButton.SetActive(visible);
        finishButton.SetActive(showFinishButton);
    }

    private void InitializeCraneHook()
    {
        SignalPerson signalPerson = craneHook.GetComponent<SignalPerson>();
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