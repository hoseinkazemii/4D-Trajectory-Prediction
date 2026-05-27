using UnityEngine;

public class LowLightHandler : MonoBehaviour
{
    private Light dirLight;  // Reference to the Light component
    private float lowLightIntensity = 0.1f;  // Lowered intensity level for a darker scenario
    private Color lowLightColor = new Color(0.2f, 0.2f, 0.5f);  // Cooler, darker color to simulate night

    private float normalIntensity;  // Store normal lighting intensity
    private Color normalColor;  // Store normal lighting color

    public Material normalSkybox;  // Store normal skybox material
    public Material lowLightSkybox;  // Low light skybox material to simulate night

    void Awake()
    {
        dirLight = GetComponent<Light>();  // Get the Light component
        normalIntensity = dirLight.intensity;  // Save the normal lighting settings
        normalColor = dirLight.color;  // Save the normal lighting color
        normalSkybox = RenderSettings.skybox;  // Save the normal skybox material
    }

    void OnEnable()
    {
        SetLowLight();  // Apply low light settings
        SetAmbientLight();  // Optionally adjust ambient light
        RenderSettings.skybox = lowLightSkybox;  // Change the skybox to the low light version
    }

    void OnDisable()
    {
        RestoreNormalLight();  // Restore normal light settings
        RestoreAmbientLight();  // Restore the default ambient light settings
        RenderSettings.skybox = normalSkybox;  // Restore the normal skybox material
    }

    private void SetLowLight()
    {
        dirLight.intensity = lowLightIntensity;  // Set low light intensity
        dirLight.color = lowLightColor;  // Set low light color
    }

    private void RestoreNormalLight()
    {
        dirLight.intensity = normalIntensity;  // Restore light intensity
        dirLight.color = normalColor;  // Restore light color
    }

    private void SetAmbientLight()
    {
        RenderSettings.ambientLight = new Color(0.1f, 0.1f, 0.2f);  // Adjust ambient light for darker scene
        RenderSettings.ambientIntensity = 0.2f;  // Set ambient intensity
    }

    private void RestoreAmbientLight()
    {
        RenderSettings.ambientLight = Color.white;  // Restore default ambient light
        RenderSettings.ambientIntensity = 1.0f;  // Restore ambient intensity
    }
}