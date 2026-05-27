using UnityEngine;
using TMPro;

public class CabinPanelManager : MonoBehaviour
{
    public TextMeshProUGUI trolleyPositionText;
    public TextMeshProUGUI hookHeightText;
    public TextMeshProUGUI radiusText;
    public TextMeshProUGUI windSpeedText;
    public TextMeshProUGUI weightText; // Add this to the inspector

    // Method to set text dynamically
    private void SetText(TextMeshProUGUI textComponent, string labelText, float value)
    {
        textComponent.text = labelText + " " + value.ToString("F1");
    }

    // Update UI elements with crane data
    public void SetTrolleyPosition(float position)
    {
        SetText(trolleyPositionText, "Trolley (ft):", position);
    }

    public void SetHook(float height)
    {
        SetText(hookHeightText, "Hook (ft):", height);
    }

    public void SetRadius(float radius)
    {
        SetText(radiusText, "Slewing Angle (ft):", radius);
    }

    public void SetWindSpeed(float windSpeed)
    {
        SetText(windSpeedText, "Wind (mi/h):", windSpeed);
    }

    public void SetLoadWeight(float weight)
    {
        SetText(weightText, "Load Weight (lbs):", weight);
    }
}




// using UnityEngine;
// using TMPro;

// public class CabinPanelManager : MonoBehaviour
// {
//     public TextMeshProUGUI trolleyPositionText;
//     public TextMeshProUGUI hookHeightText;
//     public TextMeshProUGUI radiusText;
//     public TextMeshProUGUI windSpeedText;

//     // Method to set text dynamically
//     private void SetText(TextMeshProUGUI textComponent, string labelText, float value)
//     {
//         textComponent.text = labelText + " " + value.ToString("F1");
//     }

//     // Update UI elements with crane data
//     public void SetTrolleyPosition(float position)
//     {
//         SetText(trolleyPositionText, "Trolley (ft):", position);
//     }

//     public void SetHook(float height)
//     {
//         SetText(hookHeightText, "Hook (ft):", height);
//     }

//     public void SetRadius(float radius)
//     {
//         SetText(radiusText, "Slewing Angle (ft):", radius);
//     }

//     public void SetWindSpeed(float windSpeed)
//     {
//         SetText(windSpeedText, "Wind (mi/h):", windSpeed);
//     }
// }
