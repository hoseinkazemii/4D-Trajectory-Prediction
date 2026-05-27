using UnityEngine;

public class IndustrialLightManager : MonoBehaviour
{
    // Method to activate or deactivate all child lights
    public void SetLightsActive(bool isActive)
    {
        foreach (Transform child in transform)
        {
            child.gameObject.SetActive(isActive); // Activates or deactivates each child light
        }
    }
}